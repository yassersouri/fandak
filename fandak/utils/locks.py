from oslo_concurrency import lockutils

_prefix = "fandak"
lock = lockutils.lock_with_prefix(_prefix)
lock_cleanup = lockutils.remove_external_lock_file_with_prefix(_prefix)
