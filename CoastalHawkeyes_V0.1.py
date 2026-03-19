import os
import json
import base64
import certifi
import numpy as np
import pandas as pd
import streamlit as st
import folium
import geopandas as gpd
import ee
import google_auth_oauthlib.flow
import googleapiclient.discovery
import streamlit.components.v1 as components

from folium.plugins import Draw

# =========================================================
# ENV / SSL
# =========================================================
os.environ["SSL_CERT_FILE"] = certifi.where()

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(layout="wide")

st.sidebar.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');
    .sidebar-title {
        font-size: 64px;
        font-family: 'Great Vibes', cursive;
        font-weight: 700;
    }
    </style>
    <h1 class="sidebar-title">Coastal Hawkeye</h1>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SESSION STATE INIT
# =========================================================
if "image_list" not in st.session_state:
    st.session_state["image_list"] = []
if "drive_service" not in st.session_state:
    st.session_state["drive_service"] = None
if "credentials" not in st.session_state:
    st.session_state["credentials"] = None
if "geojson_path" not in st.session_state:
    st.session_state["geojson_path"] = None

# =========================================================
# AUTH: GOOGLE EARTH ENGINE
# =========================================================
def authenticate_gee():
    """
    Earth Engine Authentication.
    """
    try:
        ee.Initialize(project="coral-style-410300")
        st.success("✅ Earth Engine already initialized.")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="coral-style-410300")
        st.success("✅ Earth Engine authenticated and initialized.")

# =========================================================
# AUTH: GOOGLE DRIVE
# =========================================================
def authenticate_google_drive():
    """
    Google Drive OAuth (local).
    """
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    CLIENT_SECRETS_FILE = r"C:/Users/zsha641/PycharmProjects/pythonProject/client_secret_742644021504-9ofiuialamam9m9rufe5tuupursnqas2.apps.googleusercontent.com.json"
    # 👉 change this path to your real json file

    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, SCOPES
    )
    credentials = flow.run_local_server(port=0)

    drive_service = googleapiclient.discovery.build(
        "drive", "v3", credentials=credentials
    )

    st.session_state["credentials"] = credentials
    st.session_state["drive_service"] = drive_service
    return drive_service

# =========================================================
# ROI: ALWAYS RETURN EXPORTABLE GEOMETRY
# =========================================================
def make_roi_geometry(location_method, point=None, polygon=None, buffer_m=2000):
    """
    Always return an ee.Geometry polygon/rectangle suitable for export.
    - Lat/Lon: buffer point then bounds() -> rectangle
    - GeoJSON: polygon.bounds()
    """
    if location_method == "Lat, Lon":
        if point is None:
            raise ValueError("Point ROI missing.")
        roi_geom = point.buffer(buffer_m).bounds()
        return roi_geom

    if location_method == "Upload GeoJSON":
        if polygon is None:
            raise ValueError("Polygon ROI missing.")
        roi_geom = polygon.bounds()
        return roi_geom

    raise ValueError("Unknown ROI method.")

# =========================================================
# MAP HELPERS
# =========================================================
def add_geojson_to_map(geojson_path, m):
    gdf = gpd.read_file(geojson_path)
    folium.GeoJson(gdf).add_to(m)
    return m

def add_gee_image_to_map(image, m, bands=["B4", "B3", "B2"]):
    """
    Add thumbnail overlay of GEE image.
    (visualization only)
    """
    url = image.getThumbURL({
        "min": 0,
        "max": 3000,
        "bands": bands,
        "dimensions": "512x512"
    })

    bounds = image.geometry().bounds().getInfo()["coordinates"][0]
    img = folium.raster_layers.ImageOverlay(
        name="GEE Image",
        image=url,
        bounds=[[bounds[1][1], bounds[0][0]], [bounds[3][1], bounds[2][0]]],
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=1
    )
    img.add_to(m)
    folium.LayerControl().add_to(m)
    return m

def folium_static(m):
    st.subheader("🗺️ Imagery visualization")
    map_html = m._repr_html_()
    components.html(
        f"""
        <div style="width:100%; height:100vh;">
            {map_html}
        </div>
        """,
        height=650
    )

# =========================================================
# THUMBNAILS UI
# =========================================================
def display_thumbnails(images):
    selected_image_ids = []
    for i, (image_id, thumb_url) in enumerate(images):
        st.image(thumb_url, caption=f"Image ID: {image_id}", use_container_width=True)
        if st.checkbox(f"Select Image {i+1}", key=f"select_{image_id}"):
            selected_image_ids.append(image_id)
    return selected_image_ids

# =========================================================
# EXPORT: FIXED VERSION (NO .geometry() BUG)
# =========================================================
def export_bands_to_drive(image, bands, roi_geom, folder, scale=10, crs=None):
    """
    Export each band as a GeoTIFF to Google Drive.
    roi_geom must be ee.Geometry (polygon/rectangle).
    """
    tasks = []
    date = image.date().format("YYYY-MM-dd").getInfo()

    for band in bands:
        band_img = image.select(band).clip(roi_geom)
        export_name = f"{date}_{band}"

        params = {
            "image": band_img,
            "description": export_name,
            "folder": folder,
            "fileNamePrefix": export_name,
            "scale": scale,
            "region": roi_geom,   # ✅ correct
            "maxPixels": 1e13
        }
        if crs:
            params["crs"] = crs

        task = ee.batch.Export.image.toDrive(**params)
        task.start()
        tasks.append(task)

    return tasks

# =========================================================
# MAIN MAP
# =========================================================
m = folium.Map(
    location=[-40.9006, 174.8860],
    zoom_start=5,
    scrollWheelZoom=False,
    tiles="cartodbpositron"
)

# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.image(
        "https://www.waikato.ac.nz/assets/Uploads/Research/Research-institutes-centres-and-groups/Institutes/eri-logo.jpg",
        width=250
    )

    st.header("Export Satellite Image from Google Earth Engine")

    # -----------------------------
    # AUTH
    # -----------------------------
    with st.expander("🔐 Authentication", expanded=True):
        if st.button("Authenticate with Google Drive"):
            try:
                authenticate_google_drive()
                st.success("✅ Authenticated with Google Drive.")
            except Exception as e:
                st.error(f"Google Drive authentication failed: {e}")

        if st.button("Authenticate with Google Earth Engine"):
            try:
                authenticate_gee()
            except Exception as e:
                st.error(f"Earth Engine authentication failed: {e}")

    # -----------------------------
    # SEARCH SETTINGS
    # -----------------------------
    with st.expander("🔎 Imagery Search", expanded=True):
        satellite = st.selectbox("Select Satellite", ["Sentinel-2"])
        # (keep only Sentinel-2 for now, because your bands are B2 B3 B4 B8)

        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        location_method = st.radio("Define ROI", ["Lat, Lon", "Upload GeoJSON"])

        point = None
        polygon = None

        if location_method == "Lat, Lon":
            location = st.text_input("Location (Lat, Lon) e.g. -36.85, 174.76")
            roi_buffer_m = st.slider("ROI half-size around point (meters)", 100, 20000, 2000, step=100)

            if location:
                try:
                    lat, lon = map(str.strip, location.split(","))
                    lat = float(lat)
                    lon = float(lon)
                    point = ee.Geometry.Point([lon, lat])
                    st.success(f"✅ Location set: lat={lat}, lon={lon}")
                except Exception as e:
                    st.error(f"Invalid Lat/Lon: {e}")

        else:
            geojson_file = st.file_uploader("Upload GeoJSON file", type=["geojson"])
            if geojson_file:
                os.makedirs("temp", exist_ok=True)
                geojson_path = os.path.join("temp", geojson_file.name)

                with open(geojson_path, "wb") as f:
                    f.write(geojson_file.getbuffer())

                try:
                    gdf = gpd.read_file(geojson_path)
                    if gdf.empty:
                        st.error("Uploaded GeoJSON is empty.")
                    else:
                        geom_type = gdf.geometry.iloc[0].geom_type

                        if geom_type == "Polygon":
                            coords = gdf.geometry.iloc[0].__geo_interface__["coordinates"]
                            polygon = ee.Geometry.Polygon(coords)
                        elif geom_type == "MultiPolygon":
                            coords = gdf.geometry.iloc[0].__geo_interface__["coordinates"][0]
                            polygon = ee.Geometry.Polygon(coords)
                        else:
                            polygon = None
                            st.error("Unsupported geometry type. Must be Polygon or MultiPolygon.")

                        if polygon:
                            st.session_state["geojson_path"] = geojson_path
                            m = add_geojson_to_map(geojson_path, m)
                            st.success("✅ GeoJSON loaded and displayed.")

                except Exception as e:
                    st.error(f"Failed to read GeoJSON: {e}")

        cloud_coverage = st.slider("Max Cloud Coverage (%)", 0, 100, 10)

        selected_bands = st.multiselect(
            "Select Bands (Sentinel-2)",
            ["B2", "B3", "B4", "B8"],
            default=["B2", "B3", "B4", "B8"]
        )
        resolution = st.slider("Export Resolution (meters)", 10, 1000, 10)

        # -----------------------------
        # FETCH DATA
        # -----------------------------
        if st.button("Fetch Data"):
            if satellite == "Sentinel-2":
                collection_id = "COPERNICUS/S2"
            else:
                collection_id = "COPERNICUS/S2"

            try:
                if location_method == "Lat, Lon":
                    if point is None:
                        st.error("Please provide a valid Lat/Lon.")
                    else:
                        roi_geom = make_roi_geometry("Lat, Lon", point=point, buffer_m=roi_buffer_m)
                        collection = (
                            ee.ImageCollection(collection_id)
                            .filterBounds(roi_geom)
                            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", cloud_coverage)
                        )

                else:
                    if polygon is None:
                        st.error("Please upload a valid GeoJSON polygon.")
                    else:
                        roi_geom = make_roi_geometry("Upload GeoJSON", polygon=polygon)
                        collection = (
                            ee.ImageCollection(collection_id)
                            .filterBounds(roi_geom)
                            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", cloud_coverage)
                        )

                if "collection" in locals():
                    images = collection.toList(collection.size()).getInfo()

                    if not images:
                        st.error("No image found for the specified parameters.")
                        st.session_state["image_list"] = []
                    else:
                        st.session_state["image_list"] = []
                        for image_info in images:
                            image_id = image_info["id"]
                            image = ee.Image(image_id)
                            thumb_url = image.getThumbURL({
                                "min": 0,
                                "max": 3000,
                                "bands": ["B4", "B3", "B2"],
                                "dimensions": "256x256"
                            })
                            st.session_state["image_list"].append((image_id, thumb_url))

                        st.success(f"✅ Found {len(st.session_state['image_list'])} images.")

            except Exception as e:
                st.error(f"Fetch failed: {e}")

        # -----------------------------
        # SHOW THUMBNAILS + EXPORT
        # -----------------------------
        if st.session_state["image_list"]:
            st.markdown("---")
            st.subheader("✅ Select images to display/export")

            selected_image_ids = display_thumbnails(st.session_state["image_list"])

            # Add selected images on map
            for image_id in selected_image_ids:
                img = ee.Image(image_id)
                m = add_gee_image_to_map(img, m)

            if selected_image_ids:
                st.success("Selected images added to the map.")

            # Drive export
            if st.session_state["drive_service"] is None:
                st.warning("Authenticate with Google Drive before exporting.")
            else:
                export_directory = st.text_input("Google Drive folder name", value="GEE_exports")

                if st.button("Export Selected Images"):
                    if not selected_image_ids:
                        st.error("No images selected.")
                    else:
                        try:
                            # build ROI geometry (polygon / rectangle)
                            if location_method == "Lat, Lon":
                                roi_geom = make_roi_geometry("Lat, Lon", point=point, buffer_m=roi_buffer_m)
                            else:
                                roi_geom = make_roi_geometry("Upload GeoJSON", polygon=polygon)

                            export_tasks = []
                            for image_id in selected_image_ids:
                                img = ee.Image(image_id)
                                tasks = export_bands_to_drive(
                                    img,
                                    selected_bands,
                                    roi_geom,
                                    export_directory,
                                    scale=resolution
                                )
                                export_tasks.extend(tasks)

                            st.success(
                                f"✅ Export tasks submitted! Check Google Drive folder: '{export_directory}'."
                            )
                            st.info("Note: Earth Engine export can take a few minutes.")

                        except Exception as e:
                            st.error(f"Export failed: {e}")

# =========================================================
# SHOW MAP
# =========================================================
folium_static(m)


# =========================================================
# COMPOSITE BANDS (UPLOAD) + OPTIONAL REFLECTANCE CORRECTION
# =========================================================
# Drop this whole block into your app (replace your current upload/correction/composite part)
# Requires: pandas as pd, numpy as np, rasterio, streamlit as st, os
# Requires your reproject_raster() function (or use the one below)

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import streamlit as st

# ---------- helpers ----------
def convert_dn_to_reflectance(dn: np.ndarray, date) -> np.ndarray:
    """
    Convert Sentinel-2 L2A DN to reflectance.
    date: python date (from st.date_input)
    """
    ref_date = pd.Timestamp("2022-01-31").date()
    dn = dn.astype("float32")

    if date < ref_date:
        refl = dn / 10000.0
    else:
        refl = (dn - 1000.0) / 10000.0

    # Optional: mask invalid values
    refl[refl < 0] = np.nan
    # If you want to cap extreme values, uncomment:
    # refl[refl > 1] = np.nan

    return refl.astype("float32")


def reproject_raster(src_path: str, dst_path: str, dst_crs: str = "EPSG:4326"):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def combine_bands_float32(band_paths, output_path):
    """
    Make a 4-band float32 GeoTIFF from single-band inputs.
    """
    with rasterio.open(band_paths[0]) as src0:
        meta = src0.meta.copy()

    meta.update(count=len(band_paths), dtype="float32")

    with rasterio.open(output_path, "w", **meta) as dst:
        for idx, p in enumerate(band_paths, start=1):
            with rasterio.open(p) as src:
                arr = src.read(1).astype("float32")
                dst.write(arr, idx)

    return output_path


# ---------- session state ----------
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {"B2": None, "B3": None, "B4": None, "B8": None}

# This is the important one: final band paths to use for compositing
if "band_paths" not in st.session_state:
    st.session_state["band_paths"] = {}

os.makedirs("temp", exist_ok=True)

# ---------- UI ----------
with st.sidebar:
    with st.expander("Composite Bands (Upload)", expanded=True):
        up_b2 = st.file_uploader("Upload Sentinel-2 B2 (Blue)", type=["tif", "tiff"], key="upload_b2")
        if up_b2: st.session_state["uploaded_files"]["B2"] = up_b2

        up_b3 = st.file_uploader("Upload Sentinel-2 B3 (Green)", type=["tif", "tiff"], key="upload_b3")
        if up_b3: st.session_state["uploaded_files"]["B3"] = up_b3

        up_b4 = st.file_uploader("Upload Sentinel-2 B4 (Red)", type=["tif", "tiff"], key="upload_b4")
        if up_b4: st.session_state["uploaded_files"]["B4"] = up_b4

        up_b8 = st.file_uploader("Upload Sentinel-2 B8 (NIR)", type=["tif", "tiff"], key="upload_b8")
        if up_b8: st.session_state["uploaded_files"]["B8"] = up_b8

        all_uploaded = all(st.session_state["uploaded_files"].values())
        if all_uploaded:
            st.success("All 4 bands uploaded.")

    if all(st.session_state["uploaded_files"].values()):

        with st.expander("Reflectance Correction", expanded=True):
            need_correction = st.checkbox("Apply reflectance correction?", value=True, key="need_corr")
            scan_date = st.date_input("Scanning Date", key="scan_date")

            if st.button("Apply / Update Band Paths", key="apply_corr_paths"):
                band_paths = {}

                for band_key, uploaded_file in st.session_state["uploaded_files"].items():
                    raw_path = os.path.join("temp", uploaded_file.name)
                    with open(raw_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if need_correction:
                        with rasterio.open(raw_path) as src:
                            dn = src.read(1)
                            profile = src.profile.copy()

                        refl = convert_dn_to_reflectance(dn, scan_date)

                        profile.update(dtype="float32", count=1, nodata=None)

                        corrected_path = os.path.join("temp", f"refl_{band_key}_{uploaded_file.name}")
                        with rasterio.open(corrected_path, "w", **profile) as dst:
                            dst.write(refl.astype("float32"), 1)

                        band_paths[band_key] = corrected_path
                    else:
                        band_paths[band_key] = raw_path

                st.session_state["band_paths"] = band_paths
                st.success("Band paths are ready for compositing (corrected if selected).")

                # Optional quick check
                b2p = st.session_state["band_paths"]["B2"]
                with rasterio.open(b2p) as src:
                    arr = src.read(1)
                st.caption(f"Quick check B2: min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f} (NaNs ignored)")

        with st.expander("Composite Bands and Export", expanded=True):
            out_path = st.text_input("Output composite path", "temp/composited_reflectance.tif", key="out_comp_path")

            do_reproject = st.checkbox("Reproject composite?", value=False, key="do_reproj")
            dst_crs = st.selectbox("Destination CRS", ["EPSG:32760", "EPSG:4326"], index=1, key="dst_crs_sel")

            if st.button("Save Composited Image", key="save_composite"):
                if not st.session_state["band_paths"]:
                    st.error("Click 'Apply / Update Band Paths' first (even if correction is OFF).")
                else:
                    bp = st.session_state["band_paths"]
                    ordered = [bp["B2"], bp["B3"], bp["B4"], bp["B8"]]

                    combined_path = combine_bands_float32(ordered, out_path)

                    final_path = combined_path
                    if do_reproject:
                        reproj_path = os.path.splitext(out_path)[0] + f"_{dst_crs.replace(':','')}.tif"
                        reproject_raster(combined_path, reproj_path, dst_crs)
                        final_path = reproj_path

                    st.success(f"✅ Composite saved: {final_path}")

                    # Optional: show basic stats for each band
                    with rasterio.open(final_path) as src:
                        stats = []
                        for i in range(1, src.count + 1):
                            a = src.read(i).astype("float32")
                            stats.append((i, float(np.nanmin(a)), float(np.nanmax(a))))
                    st.write("Band stats (min/max):", stats)



import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import Draw
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
import pandas as pd
import json
import base64
import zipfile
import os

# Functions used in the script
# Functions used in the script
def reproject_tif_to_epsg4326(input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        if src.crs is None:
            raise ValueError("The input GeoTIFF does not have a valid CRS defined.")

        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest
                )
    return output_tif


def display_raster_on_map(tif_path, rgb_indices):
    with rasterio.open(tif_path) as src:
        bands = [src.read(i).astype(float) for i in rgb_indices]
        bands = [np.where((band < 0) | (band > 0.2), np.nan, band) for band in bands]

        def normalize(array):
            array_min, array_max = np.nanmin(array), np.nanmax(array)
            return (array - array_min) / (array_max - array_min)

        bands = [normalize(band) for band in bands]

        rgb_image = np.dstack(bands) * 255
        rgb_image = rgb_image.astype(np.uint8)

        alpha_channel = np.all(np.isnan(rgb_image) == False, axis=2).astype(np.uint8) * 255
        black_pixels = np.all(rgb_image == 0, axis=2)
        alpha_channel[black_pixels] = 0

        rgba_image = np.dstack((rgb_image, alpha_channel))

        bounds = src.bounds
        bottom_left = (bounds.left, bounds.bottom)
        top_right = (bounds.right, bounds.top)

        m = folium.Map(
            location=[(bottom_left[1] + top_right[1]) / 2, (bottom_left[0] + top_right[0]) / 2],
            zoom_start=13,
            tiles='cartodbpositron'
        )

        folium.raster_layers.ImageOverlay(
            image=rgba_image,
            bounds=[[bottom_left[1], bottom_left[0]], [top_right[1], top_right[0]]],
            origin='upper',
            opacity=1.0
        ).add_to(m)

        draw = Draw(
            export=True,
            draw_options={'polyline': False, 'rectangle': False, 'circle': False, 'marker': False,
                          'circlemarker': False},
            edit_options={'edit': True}
        )
        draw.add_to(m)

        return m

def extract_reflectance(tif_path, geometry, rgb_indices):
    with rasterio.open(tif_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        out_image = out_image[rgb_indices, :, :]
        out_image = np.nan_to_num(out_image, nan=-1)
        return out_image

def geojson_download_link(geojson_data, filename='export.geojson'):
    b64 = base64.b64encode(geojson_data.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download GeoJSON</a>'
    return href

def zip_shapefile(shapefile_path):
    zip_filename = shapefile_path.replace('.shp', '.zip')
    with zipfile.ZipFile(zip_filename, 'w') as shp_zip:
        for ext in ['shp', 'shx', 'dbf', 'prj']:
            shp_zip.write(shapefile_path.replace('.shp', f'.{ext}'))
    return zip_filename

def shapefile_download_link(zip_filename, link_text="Download Shapefile"):
    with open(zip_filename, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">{link_text}</a>'
    return href

def shapefile_configuration(m):
    st.sidebar.subheader("Manual Landcover Labelling")

    uploaded_files = st.sidebar.file_uploader("Upload GeoJSON files", type=["geojson"], accept_multiple_files=True)
    if uploaded_files:
        gdfs = []
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            gdf = gpd.read_file(uploaded_file)
            gdf['id'] = i
            gdfs.append(gdf)

        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        combined_gdf.crs = gdfs[0].crs

        shapefile_path = 'combined_shapefile.shp'
        combined_gdf.to_file(shapefile_path)

        st.sidebar.success("GeoJSON files combined into one shapefile with 'id' field added.")

        id_colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#800000",  # Maroon
            "#008000",  # Dark Green
            "#000080",  # Navy
            "#808000",  # Olive
            "#800080",  # Purple
            "#008080"  # Teal
        ]

        def style_function(feature):
            id_value = feature['properties']['id']
            return {
                'fillColor': id_colors[(id_value - 1) % len(id_colors)],
                'color': id_colors[(id_value - 1) % len(id_colors)],
                'weight': 2,
                'fillOpacity': 0.5,
            }

        folium.GeoJson(combined_gdf, name="Combined GeoJSON", style_function=style_function).add_to(m)

        folium.LayerControl().add_to(m)

        return id_colors, len(uploaded_files)
    return [], 0  # Return empty list and zero if no files are uploaded
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


# Main code where the map is created
st.sidebar.header("Label Landcovers For Machine Learning")
st.sidebar.subheader("Upload the  Geotif with Composited Bands")
uploaded_file = st.sidebar.file_uploader("Choose a TIFF file", type=["tif", "tiff"])
if uploaded_file is not None:
    temp_input_tif = 'temp_input.tif'
    temp_output_tif = 'temp_output.tif'
    with open(temp_input_tif, 'wb') as out_file:
        out_file.write(uploaded_file.read())

    st.sidebar.subheader("Select Bands for RGB Display")
    rgb_bands = st.sidebar.multiselect(
        "Select three bands for RGB display (choose exactly three):",
        ['Band 2 (Blue)', 'Band 3 (Green)', 'Band 4 (Red)', 'Band 8 (NIR)'],
        default=['Band 4 (Red)', 'Band 3 (Green)', 'Band 2 (Blue)']
    )

    if len(rgb_bands) == 3:
        band_mapping = {
            'Band 2 (Blue)': 1,
            'Band 3 (Green)': 2,
            'Band 4 (Red)': 3,
            'Band 8 (NIR)': 4
        }
        rgb_indices = [band_mapping[band] for band in rgb_bands]
        try:
            st.subheader("Map for Landcover Labelling")  # Add title for the map
            reprojected_tif = reproject_tif_to_epsg4326(temp_input_tif, temp_output_tif)
            m = display_raster_on_map(reprojected_tif, rgb_indices)
            id_colors, num_files = shapefile_configuration(m)
            folium_static(m, width=800, height=600)  # Ensuring the same size for both maps

            # Display legend on the right
            if num_files > 0:
                st.sidebar.subheader("")
                legend_data = {
                    "ID": [i for i in range(1, num_files + 1)],
                    "Color": [id_colors[(i - 1) % len(id_colors)] for i in range(1, num_files + 1)]
                }
                legend_df = pd.DataFrame(legend_data)

                # Display the DataFrame with colored cells
                def color_format(val):
                    color = val
                    return f'background-color: {color}'

                styled_df = legend_df.style.applymap(color_format, subset=['Color'])
                st.dataframe(styled_df)

                zip_filename = zip_shapefile('combined_shapefile.shp')
                st.markdown(shapefile_download_link(zip_filename), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying map: {e}")
    else:
        st.sidebar.error("Please select exactly three bands for RGB display.")

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt
from skimage.measure import regionprops
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import joblib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import folium
from folium import plugins
from folium.plugins import Draw
import streamlit as st
from streamlit_folium import folium_static
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Function to reproject the classification map to EPSG:4326
def reproject_tif_to_epsg4326(input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)
    return output_tif


# Function to apply color map to the classification map
def apply_color_map(classification_map, colors):
    num_classes = int(np.nanmax(classification_map)) + 1
    cmap = ListedColormap(colors[:num_classes])
    colored_map = cmap(classification_map / np.nanmax(classification_map))
    colored_map_image = (colored_map[:, :, :3] * 255).astype(np.uint8)
    return colored_map_image, cmap


# Function to create an HTML color bar
def create_html_color_bar(cmap, vmin, vmax, colors):
    num_classes = len(colors)
    color_bar_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: auto; z-index: 1000; background-color: white; border:2px solid grey; padding: 10px;">'
    color_bar_html += '<div style="text-align: center; margin-bottom: 10px;">Density Color Bar</div>'
    color_bar_html += '<div style="display: flex; justify-content: space-between;">'
    color_bar_html += f'<span>{vmin:.2f}</span>'
    color_bar_html += '<div style="width: 240px; height: 20px; background: linear-gradient(to right, ' + ', '.join(
        [f'{color}' for color in colors[:num_classes]]) + ');"></div>'
    color_bar_html += f'<span>{vmax:.2f}</span>'
    color_bar_html += '</div></div>'
    return color_bar_html


# Function to display the raster on a Folium map
def display_raster_on_map(tif_path, labels, colors):
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        bounds = src.bounds
        bottom_left = (bounds.left, bounds.bottom)
        top_right = (bounds.right, bounds.top)
        colored_image, cmap = apply_color_map(image, colors)
        m = folium.Map(
            location=[(bottom_left[1] + top_right[1]) / 2, (bottom_left[0] + top_right[0]) / 2],
            zoom_start=13,
            tiles='cartodbpositron'
        )
        folium.raster_layers.ImageOverlay(
            image=colored_image,
            bounds=[[bottom_left[1], bottom_left[0]], [top_right[1], top_right[0]]],
            origin='upper',
            opacity=1.0
        ).add_to(m)
        draw = Draw(
            export=True,
            draw_options={'polyline': False, 'rectangle': False, 'circle': False, 'marker': False,
                          'circlemarker': False},
            edit_options={'edit': True}
        )
        draw.add_to(m)
        color_bar_html = create_html_color_bar(cmap, np.nanmin(image), np.nanmax(image), colors)
        color_bar_element = folium.Element(color_bar_html)
        m.get_root().html.add_child(color_bar_element)
        return m


# Function to sieve the classification map
def sieve_classification_map(classification_map, min_size=100):
    struct = generate_binary_structure(2, 2)
    labeled_array, num_features = label(classification_map, struct)
    props = regionprops(labeled_array)
    sieve_mask = np.zeros(classification_map.shape, dtype=bool)
    for prop in props:
        if prop.area >= min_size:
            sieve_mask[labeled_array == prop.label] = True
    sieved_map = classification_map.copy()
    sieved_map[~sieve_mask] = 0
    return sieved_map

def save_uploaded_reference_vector(uploaded_reference_files, base_dir="temp/reference_mangrove"):
    os.makedirs(base_dir, exist_ok=True)

    if not isinstance(uploaded_reference_files, list):
        uploaded_reference_files = [uploaded_reference_files]

    saved_paths = []
    for uploaded_file in uploaded_reference_files:
        save_path = os.path.join(base_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(save_path)

    primary_candidates = [p for p in saved_paths if p.lower().endswith(".gpkg") or p.lower().endswith(".geojson") or p.lower().endswith(".shp")]
    if not primary_candidates:
        raise ValueError("No readable reference vector found. Upload a GeoJSON, GPKG, or full shapefile set.")

    return primary_candidates[0]


def get_pixel_size_m(transform, crs, height, width):
    px_w = abs(transform.a)
    px_h = abs(transform.e)

    if crs is not None and getattr(crs, "is_projected", False):
        return float((px_w + px_h) / 2.0)

    center_lat = transform.f + transform.e * (height / 2.0)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(center_lat))
    px_w_m = px_w * meters_per_deg_lon
    px_h_m = px_h * meters_per_deg_lat
    return float((abs(px_w_m) + abs(px_h_m)) / 2.0)


def filter_class_by_reference_distance(classification_map, reference_vector_path, transform, crs,
                                       target_class_value=1, max_distance_m=200):
    ref_gdf = gpd.read_file(reference_vector_path)
    if ref_gdf.empty:
        raise ValueError("Reference mangrove layer is empty.")

    ref_gdf = ref_gdf[ref_gdf.geometry.notnull()].copy()
    if ref_gdf.empty:
        raise ValueError("Reference mangrove layer has no valid geometry.")

    if crs is not None:
        ref_gdf = ref_gdf.to_crs(crs)

    ref_mask = rasterize(
        [(geom, 1) for geom in ref_gdf.geometry],
        out_shape=classification_map.shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    if not np.any(ref_mask):
        raise ValueError("Reference mangrove layer does not overlap the classified TIFF extent.")

    pixel_size_m = get_pixel_size_m(transform, crs, classification_map.shape[0], classification_map.shape[1])
    distance_pixels = distance_transform_edt(~ref_mask)
    distance_m = distance_pixels * pixel_size_m

    filtered_map = classification_map.copy()
    target_mask = filtered_map == target_class_value
    remove_mask = target_mask & (distance_m > max_distance_m)
    filtered_map[remove_mask] = 0

    return filtered_map, int(np.sum(remove_mask))


def export_single_band_tif(array, output_path, transform, crs, profile, nodata_value=0):
    profile = profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw',
        transform=transform,
        crs=crs,
        nodata=nodata_value
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(rasterio.uint8), 1)


# Function to rasterize vector data
def rasterize_vector_data(vector_data, raster_meta, attribute):
    transform = raster_meta['transform']
    out_shape = (raster_meta['height'], raster_meta['width'])
    shapes = ((geom, value) for geom, value in zip(vector_data.geometry, vector_data[attribute]))
    rasterized = rasterize(shapes=shapes, out_shape=out_shape, transform=transform, fill=0, dtype='int16')
    return rasterized


def create_3d_matrix(raster_path):
    with rasterio.open(raster_path) as src:
        width = src.height
        length = src.width
        number = src.count
        container = np.ones((width, length, number), dtype=np.float32)
        for i in range(number):
            container[:, :, i] = src.read(i + 1).astype(np.float32)
    return container


def get_raster_band_array(raster_path, band_index=1):
    with rasterio.open(raster_path) as src:
        band_array = src.read(band_index)
    return band_array


def prepare_training_data(raster_path, classification_raster_path):
    container = create_3d_matrix(raster_path)
    vec = get_raster_band_array(classification_raster_path)

    # Check if the dimensions match
    if container.shape[:2] != vec.shape:
        raise ValueError(f"Shape mismatch: container shape {container.shape[:2]}, vec shape {vec.shape}")

    # Flatten the arrays where vec > 0
    fit_X = container[vec > 0]
    fit_y = vec[vec > 0]

    return fit_X, fit_y


def split_training_data(fit_X, fit_y, test_size):
    return train_test_split(fit_X, fit_y, test_size=test_size, random_state=42)


# Function to save the model to a BytesIO object and provide a download link
def save_model_download_link(model, filename="trained_model.joblib"):
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)
    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name=filename,
        mime="application/octet-stream"
    )


# Seagrass-specific functions
def calculate_ndvi(band4, band8):
    return (band8 - band4) / (band8 + band4)


def calculate_ndwi(band3, band8):
    return (band3 - band8) / (band3 + band8)


def load_model(uploaded_model):
    return joblib.load(uploaded_model)


def process_tif_files(seagrass_tif, bands_tif, dn_value):
    with rasterio.open(seagrass_tif) as src:
        seagrass_data = src.read(1)
        seagrass_mask = (seagrass_data == dn_value)
        transform = src.transform
        crs = src.crs

    with rasterio.open(bands_tif) as src:
        band2 = src.read(1)
        band3 = src.read(2)
        band4 = src.read(3)
        band8 = src.read(4)

    ndvi = calculate_ndvi(band4, band8)
    ndwi = calculate_ndwi(band3, band8)
    return band2, band3, band4, band8, ndvi, ndwi, seagrass_mask, transform, crs


def predict_seagrass_density(model, features):
    return model.predict(features)


def export_density_map(density_map, output_path, transform, crs):
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': 'nan',
        'width': density_map.shape[1],
        'height': density_map.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(density_map, 1)


def display_density_map_with_colorbar(density_map, cmap):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(density_map, cmap=cmap, vmin=np.nanmin(density_map), vmax=np.nanmax(density_map))
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Seagrass Density')
    ax.set_title("Seagrass Density Map")
    ax.axis('off')
    st.pyplot(fig)


# Additional functions for training the model
def split_data(X, y, train_size=0.8, val_size=0.1, test_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
    val_size_adjusted = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size_adjusted, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    val_r2 = r2_score(y_val, y_val_pred)
    return model, scaler, val_rmse, val_r2


def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2 = r2_score(y_test, y_test_pred)
    return test_rmse, test_r2


# Main script
if __name__ == "__main__":
    st.title("Machine Learning Model Training")

    st.sidebar.header("Analysis Tools")
    ml_operation = st.sidebar.selectbox("Select tool",
                                        ["Mangrove Reference Filter", "Landcover Detection", "Seagrass Density Estimation",
                                         "Seagrass Migration Pattern (to be established)"])

    if ml_operation == "Landcover Detection":
        st.subheader("Landcover Detection")
        model_option = st.radio("Do you want to use a trained model or train a new model?",
                                ("Use Trained Model", "Train New Model"))

        if model_option == "Use Trained Model":
            uploaded_model = st.file_uploader("Upload Trained Model", type=["joblib"], key='upload_model')
            uploaded_tif = st.file_uploader("Upload TIFF File with B2, B3, B4, B8 Bands", type=["tif"],
                                            key='upload_tif')
            output_path = st.text_input(
                "Enter the output path for the classification map (including filename and .tif extension):")

            if uploaded_model and uploaded_tif and output_path:
                st.sidebar.success("Model and TIFF file uploaded successfully.")

                model_path = os.path.join("temp", "trained_model.joblib")
                tif_path = os.path.join("temp", "uploaded_image.tif")

                os.makedirs("temp", exist_ok=True)

                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())

                with open(tif_path, "wb") as f:
                    f.write(uploaded_tif.getbuffer())

                model = joblib.load(model_path)

                with rasterio.open(tif_path) as src:
                    bands = src.read([1, 2, 3, 4])
                    transform = src.transform
                    crs = src.crs
                    profile = src.profile
                    nodata = src.nodata

                combined_image = np.stack(bands, axis=-1)
                combined_image_reshaped = combined_image.reshape(-1, 4)

                predictions = model.predict(combined_image_reshaped)
                prediction_map = predictions.reshape(combined_image.shape[:2])

                if nodata is not None:
                    prediction_map[combined_image[:, :, 0] == nodata] = 0

                sieved_map = sieve_classification_map(prediction_map, min_size=100)
                final_map = sieved_map.copy()

                def export_classification_map(prediction_map, output_path, transform, crs, profile):
                    profile.update(
                        dtype=rasterio.uint8,
                        count=1,
                        compress='lzw',
                        transform=transform,
                        crs=crs,
                        nodata=0
                    )
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(prediction_map.astype(rasterio.uint8), 1)


                export_classification_map(final_map, output_path, transform, crs, profile)
                st.sidebar.success(f"Classification map exported successfully to {output_path}")

                reprojected_tif = reproject_tif_to_epsg4326(output_path, "reprojected_classification_map.tif")

                labels = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8",
                          "Class 9", "Class 10", "Class 11", "Class 12"]

                colors = [
                    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
                    "#00FFFF", "#800000", "#008000", "#000080", "#808000",
                    "#800080", "#008080"
                ]

                st.subheader("Folium map for classification results")
                m = display_raster_on_map(reprojected_tif, labels, colors)
                folium_static(m, width=800, height=600)

                unique, counts = np.unique(final_map, return_counts=True)
                area_per_class = dict(zip(unique, counts))

                pixel_area_m2 = 10 * 10

                area_df = pd.DataFrame({
                    'Class Value': unique,
                    'Color': [
                        f'<div style="background-color:{colors[val % len(colors)]};width:25px;height:25px;"></div>' for
                        val in unique],
                    'Area (pixels)': counts,
                    'Area (m²)': counts * pixel_area_m2
                })

                st.subheader("Summary Table of Classification Map")
                st.markdown('<div style="text-align: center;">' + area_df.to_html(escape=False, index=False) + '</div>',
                            unsafe_allow_html=True)

        elif model_option == "Train New Model":
            st.info("Training a new model")
            uploaded_raster = st.file_uploader("Upload Geotif File with All Bands", type=["tif"], key='upload_raster')
            uploaded_vector_files = st.file_uploader("Upload Files (including .shp, .shx, .dbf, .prj, .cpg)",
                                                     type=["shp", "shx", "dbf", "prj", "cpg"],
                                                     accept_multiple_files=True, key='upload_vector')
            train_percent = st.slider("Training Percent", 10, 90, 70)
            test_percent = 100 - train_percent

            if uploaded_raster and uploaded_vector_files:
                st.success("Raster and vector files uploaded successfully.")

                raster_path = os.path.join("temp", "uploaded_raster.tif")

                with open(raster_path, "wb") as f:
                    f.write(uploaded_raster.getbuffer())

                vector_dir = os.path.join("temp", "vector_files")
                os.makedirs(vector_dir, exist_ok=True)

                for file in uploaded_vector_files:
                    with open(os.path.join(vector_dir, file.name), "wb") as f:
                        f.write(file.getbuffer())

                with rasterio.open(raster_path) as src:
                    bands = src.read()
                    transform = src.transform
                    raster_crs = src.crs
                    profile = src.profile
                    nodata = src.nodata

                combined_image = np.stack(bands, axis=-1)
                combined_image_reshaped = combined_image.reshape(-1, combined_image.shape[2])

                vector_files = [os.path.join(vector_dir, f.name) for f in uploaded_vector_files if
                                f.name.endswith('.shp')]
                vector_path = vector_files[0]

                vector_data = gpd.read_file(vector_path)
                st.write("Vector data fields:", vector_data.columns.tolist())
                label_column = st.selectbox("Select the label column from the vector data:",
                                            vector_data.columns.tolist())

                labels = vector_data[label_column].values

                classification_raster_path = st.text_input(
                    "Enter the output path for the classification raster (including filename and .tif extension):",
                    value="classification_ras.tif")

                if st.button("Export Classification Raster"):
                    rasterized_labels = rasterize_vector_data(vector_data, profile, label_column)

                    with rasterio.open(classification_raster_path, 'w', driver='GTiff',
                                       height=rasterized_labels.shape[0], width=rasterized_labels.shape[1], count=1,
                                       dtype=rasterized_labels.dtype, crs=raster_crs, transform=transform) as dst:
                        dst.write(rasterized_labels, 1)

                    st.success(f"Classification raster exported successfully to {classification_raster_path}")

                if classification_raster_path:
                    st.text("Set parameters for the Random Forest model:")
                    n_estimators = st.number_input("Number of estimators", min_value=1, max_value=500, value=260)
                    min_samples_split = st.number_input("Minimum samples split", min_value=2, max_value=10, value=4)
                    min_samples_leaf = st.number_input("Minimum samples leaf", min_value=1, max_value=10, value=4)
                    max_depth = st.number_input("Maximum depth", min_value=1, max_value=50, value=20)

                    if st.button("Start classification"):
                        fit_X, fit_y = prepare_training_data(raster_path, classification_raster_path)
                        fit_X_train, fit_X_test, fit_y_train, fit_y_test = split_training_data(fit_X, fit_y,
                                                                                               test_percent / 100.0)

                        st.write("Training and testing data prepared successfully.")
                        st.write(f"Training data shape: {fit_X_train.shape}")
                        st.write(f"Testing data shape: {fit_X_test.shape}")

                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_depth=max_depth,
                            random_state=42
                        )
                        model.fit(fit_X_train, fit_y_train)

                        fit_y_pred = model.predict(fit_X_test)

                        report = classification_report(fit_y_test, fit_y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()

                        conf_matrix = confusion_matrix(fit_y_test, fit_y_pred)

                        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
                        for i, accuracy in enumerate(class_accuracies):
                            report_df.loc[str(i + 1), 'accuracy'] = accuracy

                        st.subheader("Classification Report")
                        st.write(report_df)

                        save_model_download_link(model)

    elif ml_operation == "Mangrove Reference Filter":
        st.subheader("Mangrove Reference Filter")
        st.write("This is a standalone post-processing tool. Upload any classified TIFF plus a reference mangrove distribution, then remove mangrove pixels that are farther than the chosen maximum distance from the reference layer.")

        uploaded_classified_tif = st.file_uploader(
            "Upload classified TIFF",
            type=["tif", "tiff"],
            key="upload_classified_tif_for_reference_filter"
        )
        target_class_value = st.number_input(
            "Mangrove class value in the classified TIFF",
            min_value=0,
            max_value=255,
            value=1,
            step=1,
            key="sidebar_mangrove_class_value_filter"
        )
        max_distance_m = st.number_input(
            "Maximum distance from reference mangroves (m)",
            min_value=1,
            max_value=5000,
            value=200,
            step=10,
            key="sidebar_mangrove_max_distance_filter"
        )
        filtered_output_path = st.text_input(
            "Output path for filtered classified TIFF",
            value="temp/classified_mangrove_filtered.tif",
            key="sidebar_filtered_output_path"
        )
        uploaded_reference_files = st.file_uploader(
            "Upload reference mangrove distribution (GeoJSON, GPKG, or full shapefile set)",
            type=["geojson", "gpkg", "shp", "shx", "dbf", "prj", "cpg"],
            accept_multiple_files=True,
            key="sidebar_reference_mangrove_upload"
        )

        if st.button("Apply Mangrove Reference Filter", key="apply_sidebar_mangrove_reference_filter"):
            if uploaded_classified_tif is None:
                st.error("Please upload the classified TIFF.")
            elif not uploaded_reference_files:
                st.error("Please upload the reference mangrove layer.")
            elif not filtered_output_path:
                st.error("Please provide an output path for the filtered TIFF.")
            else:
                try:
                    os.makedirs("temp", exist_ok=True)
                    os.makedirs(os.path.dirname(filtered_output_path) or ".", exist_ok=True)

                    classified_tif_path = os.path.join("temp", uploaded_classified_tif.name)
                    with open(classified_tif_path, "wb") as f:
                        f.write(uploaded_classified_tif.getbuffer())

                    reference_vector_path = save_uploaded_reference_vector(uploaded_reference_files)

                    with rasterio.open(classified_tif_path) as src:
                        classification_map = src.read(1)
                        transform = src.transform
                        crs = src.crs
                        profile = src.profile

                    filtered_map, removed_pixel_count = filter_class_by_reference_distance(
                        classification_map,
                        reference_vector_path,
                        transform,
                        crs,
                        target_class_value=int(target_class_value),
                        max_distance_m=float(max_distance_m)
                    )

                    export_single_band_tif(filtered_map, filtered_output_path, transform, crs, profile, nodata_value=0)
                    st.success(
                        f"Filtered TIFF exported successfully to {filtered_output_path}. Removed {removed_pixel_count} mangrove pixels beyond {max_distance_m} m from the reference mangrove layer."
                    )

                    reprojected_tif = reproject_tif_to_epsg4326(filtered_output_path, "temp/reprojected_mangrove_reference_filtered.tif")
                    labels = [f"Class {i}" for i in range(1, 13)]
                    colors = [
                        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
                        "#00FFFF", "#800000", "#008000", "#000080", "#808000",
                        "#800080", "#008080"
                    ]
                    m = display_raster_on_map(reprojected_tif, labels, colors)
                    folium_static(m, width=800, height=600)

                    unique, counts = np.unique(filtered_map, return_counts=True)
                    area_df = pd.DataFrame({
                        'Class Value': unique,
                        'Area (pixels)': counts,
                        'Area (m²)': counts * abs(transform.a * transform.e)
                    })
                    st.dataframe(area_df)

                except Exception as e:
                    st.error(f"Mangrove reference filter failed: {e}")

    elif ml_operation == "Seagrass Density Estimation":
        st.subheader("Seagrass Density Estimation")

        # User selects whether to use a trained model or train a new model
        model_option = st.radio("Do you want to use a trained model or train a new model?",
                                ("Use Trained Model", "Train New Model"))

        if model_option == "Train New Model":
            st.subheader("Train a New Seagrass Density Estimation Model")

            uploaded_excel = st.file_uploader("Upload Excel File with Training Data", type=["xlsx"])
            output_model_path = st.text_input(
                "Enter the output path for the trained model (including filename and .joblib extension):")

            if uploaded_excel and output_model_path:
                st.success("Excel file uploaded successfully.")

                # Load Excel file
                training_data = pd.read_excel(uploaded_excel)
                X = training_data[['Band 2', 'Band 3', 'Band 4', 'Band 8', 'NDVI', 'NDWI']].values
                y = training_data['Seagrass Coverage'].values

                # Split data into training, validation, and testing sets
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

                # Train the model
                model, scaler, val_rmse, val_r2 = train_model(X_train, y_train, X_val, y_val)

                # Evaluate the model
                test_rmse, test_r2 = evaluate_model(model, scaler, X_test, y_test)

                st.write("Validation RMSE:", val_rmse)
                st.write("Validation R2 Score:", val_r2)
                st.write("Test RMSE:", test_rmse)
                st.write("Test R2 Score:", test_r2)

                # Save the model
                model_bytes = io.BytesIO()
                joblib.dump((model, scaler), model_bytes)
                model_bytes.seek(0)
                st.download_button(
                    label="Download Trained Model",
                    data=model_bytes,
                    file_name=output_model_path,
                    mime="application/octet-stream"
                )

        elif model_option == "Use Trained Model":
            # Existing functionality for using a trained model
            uploaded_model = st.file_uploader("Upload Trained SVM Model (joblib)", type=["joblib"])
            uploaded_seagrass_tif = st.file_uploader("Upload Classification map", type=["tif"], key='seagrass_tif')
            uploaded_bands_tif = st.file_uploader("Upload TIFF File with B2, B3, B4, B8 Bands", type=["tif"],
                                                  key='bands_tif')
            dn_value = st.number_input("Enter seagrass class number", value=1)
            output_density_tif = st.text_input(
                "Enter the output path for the density map (including filename and .tif extension):")

            if uploaded_model and uploaded_seagrass_tif and uploaded_bands_tif and output_density_tif:
                model_path = os.path.join("temp", "svm_model.joblib")
                seagrass_path = os.path.join("temp", "seagrass.tif")
                bands_path = os.path.join("temp", "bands.tif")

                os.makedirs("temp", exist_ok=True)

                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())

                with open(seagrass_path, "wb") as f:
                    f.write(uploaded_seagrass_tif.getbuffer())

                with open(bands_path, "wb") as f:
                    f.write(uploaded_bands_tif.getbuffer())

                model, scaler = joblib.load(model_path)
                band2, band3, band4, band8, ndvi, ndwi, seagrass_mask, transform, crs = process_tif_files(seagrass_path,
                                                                                                          bands_path,
                                                                                                          dn_value)
                band2 = np.where(band2 > 1, 0, band2)
                band3 = np.where(band3 > 1, 0, band3)
                band4 = np.where(band4 > 1, 0, band4)
                band8 = np.where(band8 > 1, 0, band8)
                ndvi = np.where(ndvi > 2, 0, ndvi)
                ndwi = np.where(ndvi > 2, 0, ndwi)

                band2 = np.where(band2 < -1, 0, band2)
                band3 = np.where(band3 < -1, 0, band3)
                band4 = np.where(band4 < -1, 0, band4)
                band8 = np.where(band8 < -1, 0, band8)
                ndvi = np.where(ndvi < -2, 0, ndvi)
                ndwi = np.where(ndvi < -2, 0, ndwi)

                features = np.stack((band2, band3, band4, band8, ndvi, ndwi), axis=-1)
                features_flat = features.reshape(-1, features.shape[-1])

                # Impute missing values
                imputer = SimpleImputer(strategy='mean')
                features_flat_imputed = imputer.fit_transform(features_flat)

                predictions = predict_seagrass_density(model, features_flat_imputed)
                seagrass_density_map = predictions.reshape(features.shape[:2])

                seagrass_density_map[~seagrass_mask] = np.nan  # Exclude other coverages
                seagrass_density_map[seagrass_density_map < 0] = np.nan  # Set values less than 0 to NaN

                # Export density map as TIFF
                export_density_map(seagrass_density_map, output_density_tif, transform, crs)
                st.sidebar.success(f"Density map exported successfully to {output_density_tif}")

                st.write("Seagrass Density Map Generated Successfully!")
                display_density_map_with_colorbar(seagrass_density_map, LinearSegmentedColormap.from_list("greens",
                                                                                                          ["#e0f3db",
                                                                                                           "#a8ddb5",
                                                                                                           "#43a2ca"]))


st.title("__________________________________________________________________________")

st.markdown(
"""
    </style>
    """
    """
    <footer style="position: relative; bottom: 0; width: 100%; text-align: center;">
    <p>   <p>
    <p>   <p>
    <p>   <p>
    <p>   <p>
    <p>   <p>
    <p> \n <p>
    <p>If you have any questions, feel free to <a href="mailto:z.shao@waikato.ac.nz">email me</a>.</p>
    <p>Follow me on the following social media.<p>
    <p> 
        <a href="https://nz.linkedin.com/in/zhanchao-shao-a56716123" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" style="width:24px;height:24px;">
        </a>   
        <a href="https://x.com/ZhanchaoShao" target="_blank">
            <img   src="https://cdn-icons-png.flaticon.com/512/733/733579.png" alt="Twitter" style="width:24px;height:24px;">
        </a>
    </p>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <footer style="position: relative; bottom: 0; width: 100%; text-align: center;">
        <div style="display: flex; justify-content: center; align-items: center;">
            <a href="https://www.waikato.ac.nz" target="_blank" style="margin-right: 20px;">
                <img src="https://profiles.waikato.ac.nz/branding/large-logo.svg" alt="University of Waikato" style="height: 50px;">
            </a>
            <a href="https://www.auckland.ac.nz" target="_blank">
                <img src="https://profiles.auckland.ac.nz/branding/large-logo.svg" alt="University of Auckland" style="height: 30px;">
            </a>
        </div>
    </footer>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url(https://images.squarespace-cdn.com/content/v1/58870b02f7e0ab5d5e958437/1711530417923-YVJUB2P9JVN2T51FVFMY/00019-photographers-tauranga-papamoa.jpg?format=2500w);
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
