import time
import asyncio
from datetime import datetime, timedelta
from .utils import logger


def get_minute_slot():
    current_time = time.time()
    dt_object = datetime.fromtimestamp(current_time)
    total_minutes_since_midnight = dt_object.hour * 60 + dt_object.minute
    return total_minutes_since_midnight


class RPM:

    def __init__(self, rpm: int = 1000):
        self.rpm = rpm
        self.record = {'rpm_slot': get_minute_slot(), 'counter': 0}

    async def wait(self, silent=False):
        current = time.time()
        dt_object = datetime.fromtimestamp(current)
        minute_slot = get_minute_slot()

        if self.record['rpm_slot'] == minute_slot:
            # check RPM exceed
            if self.record['counter'] >= self.rpm:
                # wait until next minute
                next_minute = dt_object.replace(
                    second=0, microsecond=0) + timedelta(minutes=1)
                _next = next_minute.timestamp()
                sleep_time = abs(_next - current)
                if not silent:
                    logger.info('RPM sleep %s', sleep_time)
                await asyncio.sleep(sleep_time)

                self.record = {
                    'rpm_slot': get_minute_slot(),
                    'counter': 0
                }
        else:
            self.record = {'rpm_slot': get_minute_slot(), 'counter': 0}
        self.record['counter'] += 1

        if not silent:
            logger.debug(self.record)


class TPM:

    def __init__(self, tpm: int = 20000):
        self.tpm = tpm
        self.record = {'tpm_slot': get_minute_slot(), 'counter': 0}

    async def wait(self, token_count, silent=False):
        current = time.time()
        dt_object = datetime.fromtimestamp(current)
        minute_slot = get_minute_slot()

        # get next slot, skip
        if self.record['tpm_slot'] != minute_slot:
            self.record = {'tpm_slot': minute_slot, 'counter': token_count}
            return

        # check RPM exceed
        self.record['counter'] += token_count
        if self.record['counter'] > self.tpm:
            # wait until next minute
            next_minute = dt_object.replace(
                second=0, microsecond=0) + timedelta(minutes=1)
            _next = next_minute.timestamp()
            sleep_time = abs(_next - current)
            logger.info('TPM sleep %s', sleep_time)
            await asyncio.sleep(sleep_time)

            self.record = {
                'tpm_slot': get_minute_slot(),
                'counter': token_count
            }

        if not silent:
            logger.debug(self.record)
