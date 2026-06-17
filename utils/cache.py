import sqlite3
from pathlib import Path
import os
import io
from collections import defaultdict, OrderedDict

import torch


class Cache:
    def __init__(self, path: str, fingerprint: str, shard_size_gb=1):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.metadata_db = self.path / 'metadata.db'
        self.shard_size_gb = shard_size_gb
        os.makedirs(self.path, exist_ok=True)

        self.init()


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        assert isinstance(idx, int)
        shard_id, shard_index = self.items[idx]
        offset, size = self.shard_metadata[shard_id][shard_index]
        if shard_id in self.open_files:
            self.open_files.move_to_end(shard_id)
        else:
            if len(self.open_files) >= 64:
                _, old_f = self.open_files.popitem(last=False)
                old_f.close()
            self.open_files[shard_id] = open(self.path / f'shard_{shard_id}.bin', 'rb')
        f = self.open_files[shard_id]
        f.seek(offset)
        byte_string = f.read(size)
        buffer = io.BytesIO(byte_string)
        try:
            item = torch.load(buffer, map_location='cpu', weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only parameter
            item = torch.load(buffer, map_location='cpu')
        # Data corruption or disk errors are not caught — they propagate up
        # so callers can decide whether to retry or clear the cache.
        return item


    def init(self):
        print('[CACHE] Initializing')
        # create database
        self.con = sqlite3.connect(self.metadata_db, autocommit=False)

        # check fingerprint, clear cache if different
        self.con.execute('CREATE TABLE IF NOT EXISTS fingerprint(value)')
        existing_fingerprint = self.con.execute('SELECT value FROM fingerprint').fetchone()
        if existing_fingerprint is not None:
            existing_fingerprint = existing_fingerprint[0]
            print(f'[CACHE] Existing cache has fingerprint {existing_fingerprint}')
            if self.fingerprint != existing_fingerprint:
                print('[CACHE] Fingerprint changed, deleting existing cache files')
                self.clear()
                return
        else:
            print(f'[CACHE] Storing new fingerprint: {self.fingerprint}')
            self.con.execute('INSERT INTO fingerprint VALUES(?)', (self.fingerprint,))

        # items table, current length, next shard index
        self.con.execute('CREATE TABLE IF NOT EXISTS items(shard INT, shard_index INT, PRIMARY KEY(shard, shard_index))')

        # migration: detect old table without primary key and rebuild
        cols = self.con.execute('PRAGMA table_info(items)').fetchall()
        has_pk = any(col[5] for col in cols)  # col[5] = pk flag
        if cols and not has_pk:
            print('[CACHE] Migrating items table to add primary key')
            self.con.execute('ALTER TABLE items RENAME TO items_old')
            self.con.execute('CREATE TABLE items(shard INT, shard_index INT, PRIMARY KEY(shard, shard_index))')
            self.con.execute('INSERT OR IGNORE INTO items SELECT * FROM items_old')
            self.con.execute('DROP TABLE items_old')
            self.con.commit()
        self.items = self.con.execute('SELECT shard, shard_index FROM items').fetchall() or []
        max_existing_shard = -1
        for shard, _ in self.items:
            max_existing_shard = max(max_existing_shard, shard)
        self.shard = max_existing_shard + 1  # current shard to write to
        self.shard_file = None
        print(f'[CACHE] Existing cache length: {len(self)}')

        # shard metadata
        self.shard_metadata = defaultdict(list)
        for table_name, in self.con.execute('SELECT name FROM sqlite_master').fetchall():
            if table_name.startswith('shard_'):
                shard_id = int(table_name.split('_')[-1])
                for entry in self.con.execute(f'SELECT offset, size FROM {table_name}').fetchall():
                    self.shard_metadata[shard_id].append(entry)
        # LRU-ordered dict caps open file handles at 64, evicting least recently used
        self.open_files = OrderedDict()

        # commit
        self.con.commit()


    def clear(self):
        '''Deletes all cache files from disk. Calls init() again.'''
        self.con.close()
        # Phase 1: delete database first. If this fails, abort — bin files are
        # left behind for manual cleanup rather than risking an inconsistent state.
        try:
            os.remove(self.metadata_db)
        except OSError as e:
            print(f'[CACHE] FATAL: could not remove database {self.metadata_db}: {e}')
            raise
        # Phase 2: delete bin files individually. Partial failures are warned
        # but do not abort — leftover bin files are harmless.
        for bin_path in self.path.glob('*.bin'):
            try:
                os.remove(bin_path)
            except OSError as e:
                print(f'[CACHE] Warning: could not remove {bin_path}: {e}')
        self.init()


    def create_new_shard(self):
        self.shard_file = open(self.path / f'shard_{self.shard}.bin', 'wb')
        self.shard_table = f'shard_{self.shard}'
        print(f'[CACHE] Creating new shard: {self.shard_table}')
        self.con.execute(f'CREATE TABLE {self.shard_table}(offset, size)')
        self.shard_index = 0
        self.offset = 0


    def finalize_current_shard(self):
        if self.shard_file is None:
            # no-op if already finalized
            return
        self.shard_file.close()
        self.shard_file = None
        self.shard += 1
        self.con.commit()


    def warmup(self):
        """Pre-read all shard files into the OS page cache for faster subsequent reads."""
        for shard_id in range(self.shard + 1):
            path = self.path / f'shard_{shard_id}.bin'
            if path.exists():
                with open(path, 'rb') as f:
                    while f.read(1024 * 1024):
                        pass

    def init_readonly(self):
        """Reopen SQLite and file handles in read-only mode for use in DataLoader workers.

        After fork(), the inherited SQLite connection and file handles are shared
        across worker processes, which can cause lock contention or seek races.
        Call this from worker_init_fn to get a private read-only connection.

        Called from _worker_init_fn after DataLoader fork. Each worker gets
        its own copy of the Cache object (copy-on-write), so there is no
        cross-worker contention on self.con. Read-only URI mode avoids
        SQLite write lock contention entirely.
        """
        self.con.close()
        self.con = sqlite3.connect(
            f'file:{self.metadata_db}?mode=ro', uri=True
        )
        self.open_files = OrderedDict()  # maintain LRU eviction behavior from init()


    def add(self, item):
        if self.shard_file is None:
            self.create_new_shard()
        buffer = io.BytesIO()
        torch.save(item, buffer)
        bytes_view = buffer.getbuffer()
        self.shard_file.write(bytes_view)

        # update items metadata
        item = (self.shard, self.shard_index)
        self.items.append(item)
        self.con.execute('INSERT INTO items VALUES(?, ?)', item)
        self.shard_index += 1

        # periodic commit to avoid WAL growth and data loss on crash
        if self.shard_index % 1000 == 0 or (self.shard_file is not None and self.shard_file.tell() > 500_000_000):
            self.con.commit()

        # update shard metadata
        size = len(bytes_view)
        entry = (self.offset, size)
        self.shard_metadata[self.shard].append(entry)
        self.con.execute(f'INSERT INTO {self.shard_table} VALUES (?, ?)', entry)
        self.offset += size

        # create new shard when existing one is large enough
        current_size_gb = self.shard_file.tell() / 1_000_000_000
        if current_size_gb >= self.shard_size_gb:
            self.finalize_current_shard()


# for testing
if __name__ == '__main__':
    cache = Cache('/home/anon/tmp/cache_test', 'foo', shard_size_gb=0.001)

    tensor = torch.zeros((100_000,))
    for _ in range(10):
        cache.add({'key1': tensor})
    cache.finalize_current_shard()

    print(cache[0])
    print(cache[1])
