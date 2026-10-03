"""Shim for the `libusb-package` PyPI module, which has no win_arm64 wheel.

cflib imports `libusb_package` at module level (cflib/drivers/crazyradio.py), so without
it `import cflib.crtp` fails even for non-USB links. This shim offers the same API and
defers to pyusb's normal backend discovery (libusb-1.0.dll on PATH, if any). With no
libusb DLL present, USB scans simply find no devices; BLE is unaffected.
"""
import usb.backend.libusb1
import usb.core


def get_libusb1_backend():
    return usb.backend.libusb1.get_backend()


def find(*args, **kwargs):
    kwargs.setdefault("backend", get_libusb1_backend())
    return usb.core.find(*args, **kwargs)


def get_library_path():
    return None


def find_library(candidate=None):
    return None
