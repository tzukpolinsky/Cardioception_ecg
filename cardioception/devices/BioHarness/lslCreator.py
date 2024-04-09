import asyncio
import datetime
import logging
import threading
import time
from time import sleep

import numpy as np
import pandas as pd
import pylsl
import neurokit2 as nk

from cardioception.devices.BioHarness import BioHarness
from cardioception.devices.BioHarness.protocol import ECGWaveformMessage, BreathingWaveformMessage, \
    Accelerometer100MgWaveformMessage, AccelerometerWaveformMessage, RtoRMessage, get_unit

logger = logging.getLogger(__name__)


class BioHarnessLslCreator:
    def __init__(self, address, port, timeout, modalities=None, stream_prefix='Zephyr',
                 local_time=None):
        if modalities is None:
            modalities = ['ECG', 'Respiration']
        self.enablers = {
            'ECG': self.enable_ecg,
            'Respiration': self.enable_respiration,
            'Accel100mg': self.enable_accel100mg,
            'Accel': self.enable_accel,
            'RtoR': self.enable_rtor,
            'Events': self.enable_events,
            'Summary': self.enable_summary,
            'General': self.enable_general,
        }
        self.link = None
        self.modalities = modalities
        self.port = port
        self.address = address
        self.timeout = timeout
        self.local_time = local_time
        self.stream_prefix = stream_prefix
        loop = asyncio.new_event_loop()
        asyncio.run_coroutine_threadsafe(self.init(), loop)
        self.wait = True
        thread = threading.Thread(target=self.start_async_loop, args=(loop,))
        thread.start()
        while self.wait:
            sleep(0.1)

    def start_async_loop(self, l):
        asyncio.set_event_loop(l)
        try:
            l.run_forever()
        except KeyboardInterrupt:
            logger.info("Ctrl-C pressed.")
        finally:
            if self.link:
                # noinspection PyUnresolvedReferences
                link.shutdown()
            l.close()

    async def init(self):
        try:
            # set up logging
            unknown = set(self.modalities) - set(self.enablers.keys())
            if unknown:
                raise ValueError(f"Unknown modalities to stream: {unknown}")

            # connect to bioharness
            self.link = BioHarness(self.address, port=int(self.port), timeout=int(self.timeout))
            infos = await self.link.get_infos()
            info_str = '\n'.join([f' * {k}: {v}' for k, v in infos.items()])
            logger.info(f"Device info is:\n{info_str}")
            id_prefix = infos['serial']

            # enable various kinds of streams and install handlers
            logger.info("Enabling streams...")
            for mod in self.modalities:
                logger.info(f"  enabling {mod}...")
                enabler = self.enablers[mod]
                if mod == 'Events':
                    await enabler(self.link, name_prefix=self.stream_prefix, id_prefix=id_prefix,
                                  local_time=self.local_time)
                else:
                    await enabler(self.link, name_prefix=self.stream_prefix, id_prefix=id_prefix)

            logger.info('Now streaming...')
            self.wait = False
        except SystemExit:
            asyncio.get_event_loop().stop()
        except TimeoutError as e:
            logger.error(f"Operation timed out: {e}")
            asyncio.get_event_loop().stop()
        except Exception as e:
            logger.exception(e)
            asyncio.get_event_loop().stop()

    def add_manufacturer(self, desc):
        """Add manufacturer into to a stream's desc"""
        acq = desc.append_child('acquisition')
        acq.append_child_value('manufacturer', 'Medtronic')
        acq.append_child_value('model', 'Zephyr BioHarness')

    async def enable_ecg(self, link, name_prefix, id_prefix):
        """Enable the ECG data stream. This is the raw ECG waveform."""
        info = pylsl.StreamInfo(name_prefix + 'ECG', 'ECG', 1,
                                nominal_srate=ECGWaveformMessage.srate,
                                source_id=id_prefix + '-ECG')
        desc = info.desc()
        chn = desc.append_child('channels').append_child('channel')
        chn.append_child_value('label', 'ECG1')
        chn.append_child_value('type', 'ECG')
        chn.append_child_value('unit', 'millivolts')
        self.add_manufacturer(desc)
        outlet = pylsl.StreamOutlet(info)

        def on_ecg(msg):
            outlet.push_chunk([[v] for v in msg.waveform])

        await link.toggle_ecg(on_ecg)

    # noinspection PyUnusedLocal
    async def enable_respiration(self, link, name_prefix, id_prefix):
        """Enable the respiration data stream. This is the raw respiration (chest
        expansion) waveform."""
        info = pylsl.StreamInfo(name_prefix + 'Resp', 'Respiration', 1,
                                nominal_srate=BreathingWaveformMessage.srate,
                                source_id=id_prefix + '-Resp')
        desc = info.desc()
        chn = desc.append_child('channels').append_child('channel')
        chn.append_child_value('label', 'Respiration')
        chn.append_child_value('type', 'EXG')
        chn.append_child_value('unit', 'unnormalized')
        self.add_manufacturer(desc)
        outlet = pylsl.StreamOutlet(info)

        def on_breathing(msg):
            outlet.push_chunk([[v] for v in msg.waveform])

        await link.toggle_breathing(on_breathing)

    # noinspection PyUnusedLocal
    async def enable_accel100mg(self, link, name_prefix, id_prefix):
        """Enable the accelerometer data stream. This is a 3-channel stream in units
        of 1 g (earth gravity)."""
        info = pylsl.StreamInfo(name_prefix + 'Accel100mg', 'Accel100mg', 3,
                                nominal_srate=Accelerometer100MgWaveformMessage.srate,
                                source_id=id_prefix + '-Accel100mg')
        desc = info.desc()
        chns = desc.append_child('channels')
        for lab in ['X', 'Y', 'Z']:
            chn = chns.append_child('channel')
            chn.append_child_value('label', lab)
            chn.append_child_value('unit', 'g')
            chn.append_child_value('type', 'Acceleration' + lab)
        self.add_manufacturer(desc)
        outlet = pylsl.StreamOutlet(info)

        def on_accel100mg(msg):
            outlet.push_chunk([[x, y, z] for x, y, z in zip(msg.accel_x, msg.accel_y, msg.accel_z)])

        await link.toggle_accel100mg(on_accel100mg)

    # noinspection PyUnusedLocal
    async def enable_accel(self, link, name_prefix, id_prefix):
        """Enable the regular accelerometer data stream. This is a 3-channel stream
        with slightly higher res than accel100mg (I believe around 2x), but """
        info = pylsl.StreamInfo(name_prefix + 'Accel', 'Accel', 3,
                                nominal_srate=AccelerometerWaveformMessage.srate,
                                source_id=id_prefix + '-Accel')
        desc = info.desc()
        chns = desc.append_child('channels')
        for lab in ['X', 'Y', 'Z']:
            chn = chns.append_child('channel')
            chn.append_child_value('label', lab)
            chn.append_child_value('type', 'Acceleration' + lab)
            chn.append_child_value('unit', 'unnormalized')
        self.add_manufacturer(desc)
        outlet = pylsl.StreamOutlet(info)

        def on_accel(msg):
            outlet.push_chunk([[x, y, z] for x, y, z in zip(msg.accel_x, msg.accel_y, msg.accel_z)])

        await link.toggle_accel(on_accel)

    # noinspection PyUnusedLocal
    async def enable_rtor(self, link, name_prefix, id_prefix):
        """Enable the RR interval data stream. This has the interval between the
        most recent two ECG R-waves, in ms (held constant until the next R-peak),
        and the sign of the reading alternates with each new R peak."""
        info = pylsl.StreamInfo(name_prefix + 'RtoR', 'RtoR', 1,
                                nominal_srate=RtoRMessage.srate,
                                source_id=id_prefix + '-RtoR')
        desc = info.desc()
        chn = desc.append_child('channels').append_child('channel')
        chn.append_child_value('label', 'RtoR')
        chn.append_child_value('unit', 'milliseconds')
        chn.append_child_value('type', 'Misc')

        self.add_manufacturer(desc)
        outlet = pylsl.StreamOutlet(info)

        def on_rtor(msg):
            outlet.push_chunk([[v] for v in msg.waveform])

        await link.toggle_rtor(on_rtor)

    async def enable_events(self, link, name_prefix, id_prefix, local_time='1'):
        """Enable the events data stream. This has a few system events like button
        pressed, battery low, worn status changed."""
        info = pylsl.StreamInfo(name_prefix + 'Events', 'Events', 1,
                                nominal_srate=0,
                                channel_format=pylsl.cf_string,
                                source_id=id_prefix + '-Events')
        outlet = pylsl.StreamOutlet(info)

        def on_event(msg):
            if local_time == '1':
                stamp = datetime.datetime.fromtimestamp(msg.stamp)
            else:
                stamp = datetime.datetime.utcfromtimestamp(msg.stamp)
            timestr = stamp.strftime('%Y-%m-%d %H:%M:%S')
            event_str = f'{msg.event_string}/{msg.event_data}@{timestr}'
            outlet.push_sample([event_str])
            logger.debug(f'event detected: {event_str}')

        await link.toggle_events(on_event)

    # noinspection PyUnusedLocal
    async def enable_summary(self, link, name_prefix, id_prefix):
        """Enable the summary data stream. This has most of the derived data
        channels in it."""
        # we're delaying creation of these objects until we got data since we don't
        # know in advance if we're getting summary packet V2 or V3
        info, outlet = None, None

        def on_summary(msg):
            nonlocal info, outlet
            content = msg.as_dict()
            if info is None:
                info = pylsl.StreamInfo(name_prefix + 'Summary', 'Summary', len(content),
                                        nominal_srate=1,
                                        channel_format=pylsl.cf_float32,
                                        source_id=id_prefix + '-Summary')
                desc = info.desc()
                self.add_manufacturer(desc)
                chns = desc.append_child('channels')
                for key in content:
                    chn = chns.append_child('channel')
                    chn.append_child_value('label', key)
                    unit = get_unit(key)
                    if unit is not None:
                        chn.append_child_value('unit', unit)
                outlet = pylsl.StreamOutlet(info)
            outlet.push_sample(list(content.values()))

        await link.toggle_summary(on_summary)

    # noinspection PyUnusedLocal
    async def enable_general(self, link, name_prefix, id_prefix):
        """Enable the general data stream. This has summary metrics, but fewer than
        the summary stream, plus a handful of less-useful channels."""
        # we're delaying creation of these objects until we got data since we're
        # deriving the channel count and channel labels from the data packet
        info, outlet = None, None

        def on_general(msg):
            nonlocal info, outlet
            content = msg.as_dict()
            if info is None:
                info = pylsl.StreamInfo(name_prefix + 'General', 'General', len(content),
                                        nominal_srate=1,
                                        channel_format=pylsl.cf_float32,
                                        source_id=id_prefix + '-General')
                desc = info.desc()
                self.add_manufacturer(desc)
                chns = desc.append_child('channels')
                for key in content:
                    chn = chns.append_child('channel')
                    chn.append_child_value('label', key)
                    unit = get_unit(key)
                    if unit is not None:
                        chn.append_child_value('unit', unit)
                outlet = pylsl.StreamOutlet(info)
            outlet.push_sample(list(content.values()))

        await link.toggle_general(on_general)


class BioHarnessTask(BioHarnessLslCreator):
    def __init__(self, address, port, timeout, modalities=None, stream_prefix='Zephyr',
                 local_time=None):
        super().__init__(address, port, timeout, modalities, stream_prefix, local_time)
        self.signals = {}
        self.all_recording = []
        self.timestamps = []
        self.recording = []
        self.channels = {"Channel_0": []}
        for mod in self.modalities:
            streams = pylsl.resolve_stream('type', mod)
            if len(streams) > 0:
                self.signals[mod] = pylsl.StreamInlet(streams[0])

    def read(self, duration, mod='ECG'):
        self.recording = []
        self.readInWaiting()  # clear the stack of lsl before reading
        start_time = time.time()
        while time.time() - start_time <= duration:
            samples, timestamps = self.signals[mod].pull_chunk()
            self.timestamps = self.timestamps + timestamps
            self.recording = self.recording + list(np.array(samples))
            self.channels["Channel_0"].append(0)
        self.all_recording = self.all_recording + self.recording
        return self

    def readInWaiting(self, mod='ECG'):
        self.signals[mod].pull_chunk()  # clear the stack of lsl before reading
        self.channels["Channel_0"].append(0)

    def save(self, fname):
        saveList = np.array([np.array(self.all_recording).flatten(), np.array(self.timestamps).flatten()])
        if fname.endswith(".txt"):
            colnames = ["signal", "time"]
            pd.DataFrame(saveList.T, columns=colnames).to_csv(
                fname, index=False
            )
        else:
            np.save(fname, saveList)

    def setup(self):
        self.readInWaiting()
        return self

    def get_peaks(self, duration=5.0, sampling_rate=250):
        signal = (
            self.read(duration=duration).recording  # noqa
        )
        if len(signal) == 0:
            print("cant get peaks, len zero")
            return [], np.array([])
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="neurokit")
        signals, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="neurokit")
        return signal, np.array(signals["ECG_R_Peaks"]).astype(bool)
