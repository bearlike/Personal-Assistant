#!/usr/bin/env python3
import time


def get_unique_timestamp():
    # Get the number of seconds since epoch (Jan 1, 1970) as a float
    current_timestamp = int(time.time())
    # Convert it to string for uniqueness and consistency
    unique_timestamp = str(current_timestamp)
    # Return the integer version of this string timestamp
    return int(''.join(str(x) for x in map(int, unique_timestamp)))
