"""
Contains functions for parsing geo metadata from images
"""

import datetime as dt
from collections import OrderedDict
from PIL.ExifTags import TAGS, GPSTAGS


def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    info = image._getexif()
    if not info:
        return {}
    exif_data = {TAGS.get(tag, tag): value for tag, value in info.items()}

    def is_fraction(val):
        return isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], int) and isinstance(val[1], int)

    def frac_to_dec(frac):
        return float(frac[0]) / float(frac[1])

    if "GPSInfo" in exif_data:
        gpsinfo = {GPSTAGS.get(t, t): v for t, v in exif_data["GPSInfo"].items()}
        for tag, value in gpsinfo.items():
            if is_fraction(value):
                gpsinfo[tag] = frac_to_dec(value)
            elif all(is_fraction(x) for x in value):
                gpsinfo[tag] = tuple(map(frac_to_dec, value))
        exif_data["GPSInfo"] = gpsinfo
    return exif_data


def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data"""
    lat = None
    lon = None
    gps_info = exif_data.get("GPSInfo")

    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    if gps_info:
        gps_latitude = gps_info.get("GPSLatitude")
        gps_latitude_ref = gps_info.get("GPSLatitudeRef")
        gps_longitude = gps_info.get("GPSLongitude")
        gps_longitude_ref = gps_info.get("GPSLongitudeRef")

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_to_degrees(gps_latitude)
            if gps_latitude_ref != "N":
                lat = -lat

            lon = convert_to_degrees(gps_longitude)
            if gps_longitude_ref != "E":
                lon = -lon

    return lat, lon


def get_gps_datetime(exif_data):
    """Returns the timestamp, if available, from the provided exif_data"""
    if "GPSInfo" not in exif_data:
        return None
    gps_info = exif_data["GPSInfo"]
    date_str = gps_info.get("GPSDateStamp")
    time = gps_info.get("GPSTimeStamp")
    if not date_str or not time:
        return None
    date = map(int, date_str.split(":"))
    timestamp = [*date, *map(int, time)]
    timestamp += [int((time[2] % 1) * 1e6)]  # microseconds
    return dt.datetime(*timestamp)


def clean_gps_info(exif_data):
    """Return GPS EXIF info in a more convenient format from the provided exif_data"""
    gps_info = exif_data["GPSInfo"]
    cleaned = OrderedDict()
    cleaned["Latitude"], cleaned["Longitude"] = get_lat_lon(exif_data)
    cleaned["Altitude"] = gps_info.get("GPSAltitude")
    cleaned["Speed"] = gps_info.get("GPSSpeed")
    cleaned["SpeedRef"] = gps_info.get("GPSSpeedRef")
    cleaned["Track"] = gps_info.get("GPSTrack")
    cleaned["TrackRef"] = gps_info.get("GPSTrackRef")
    cleaned["TimeStamp"] = get_gps_datetime(exif_data)
    return cleaned


def get_gps_info(img):
    return clean_gps_info(exif_data=get_exif_data(img))
