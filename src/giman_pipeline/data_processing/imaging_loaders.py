"""Imaging data loaders for XML metadata and DICOM file processing.

This module provides functions to parse XML metadata files that describe
DICOM image collections, extract relevant information, and prepare it
for integration with the tabular PPMI data pipeline.

Key Functions:
    - parse_xml_metadata: Parse individual XML files for imaging metadata
    - load_all_xml_metadata: Batch load all XML files from a directory
    - map_visit_identifiers: Map imaging visit IDs to standard EVENT_ID format
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_xml_metadata(xml_file_path: str | Path) -> dict[str, str] | None:
    """Parse a single XML metadata file to extract DICOM imaging information.

    Args:
        xml_file_path: Path to the XML metadata file

    Returns:
        Dictionary containing extracted metadata, or None if parsing fails

    Example:
        >>> metadata = parse_xml_metadata("scan_001.xml")
        >>> print(metadata['subjectIdentifier'])
        '3001'
    """
    xml_path = Path(xml_file_path)

    if not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        return None

    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Initialize metadata dictionary
        metadata = {
            "xml_filename": xml_path.name,
            "subjectIdentifier": None,
            "visitIdentifier": None,
            "modality": None,
            "dateAcquired": None,
            "imageUID": None,
            "seriesDescription": None,
            "manufacturer": None,
            "fieldStrength": None,
            "protocolName": None,
            "sliceThickness": None,
            "repetitionTime": None,
            "echoTime": None,
        }

        # Extract key metadata fields
        # Note: These XPath expressions may need adjustment based on actual XML structure
        subject_elem = root.find(".//subjectIdentifier")
        if subject_elem is None:
            subject_elem = root.find(".//subject_id")
        if subject_elem is not None:
            metadata["subjectIdentifier"] = subject_elem.text

        visit_elem = root.find(".//visitIdentifier")
        if visit_elem is None:
            visit_elem = root.find(".//visit_id")
        if visit_elem is not None:
            metadata["visitIdentifier"] = visit_elem.text

        modality_elem = root.find(".//modality")
        if modality_elem is None:
            modality_elem = root.find(".//Modality")
        if modality_elem is not None:
            metadata["modality"] = modality_elem.text

        date_elem = root.find(".//dateAcquired")
        if date_elem is None:
            date_elem = root.find(".//StudyDate")
        if date_elem is not None:
            metadata["dateAcquired"] = date_elem.text

        uid_elem = root.find(".//imageUID")
        if uid_elem is None:
            uid_elem = root.find(".//SeriesInstanceUID")
        if uid_elem is not None:
            metadata["imageUID"] = uid_elem.text

        # Additional imaging parameters
        series_desc_elem = root.find(".//seriesDescription")
        if series_desc_elem is None:
            series_desc_elem = root.find(".//SeriesDescription")
        if series_desc_elem is not None:
            metadata["seriesDescription"] = series_desc_elem.text

        manufacturer_elem = root.find(".//manufacturer")
        if manufacturer_elem is None:
            manufacturer_elem = root.find(".//Manufacturer")
        if manufacturer_elem is not None:
            metadata["manufacturer"] = manufacturer_elem.text

        field_strength_elem = root.find(".//fieldStrength")
        if field_strength_elem is None:
            field_strength_elem = root.find(".//MagneticFieldStrength")
        if field_strength_elem is not None:
            metadata["fieldStrength"] = field_strength_elem.text

        protocol_elem = root.find(".//protocolName")
        if protocol_elem is None:
            protocol_elem = root.find(".//ProtocolName")
        if protocol_elem is not None:
            metadata["protocolName"] = protocol_elem.text

        slice_thick_elem = root.find(".//sliceThickness")
        if slice_thick_elem is None:
            slice_thick_elem = root.find(".//SliceThickness")
        if slice_thick_elem is not None:
            metadata["sliceThickness"] = slice_thick_elem.text

        tr_elem = root.find(".//repetitionTime")
        if tr_elem is None:
            tr_elem = root.find(".//RepetitionTime")
        if tr_elem is not None:
            metadata["repetitionTime"] = tr_elem.text

        te_elem = root.find(".//echoTime")
        if te_elem is None:
            te_elem = root.find(".//EchoTime")
        if te_elem is not None:
            metadata["echoTime"] = te_elem.text

        logger.info(f"Successfully parsed XML metadata from {xml_path.name}")
        return metadata

    except ET.ParseError as e:
        logger.error(f"XML parsing error in {xml_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {xml_path}: {e}")
        return None


def map_visit_identifiers(visit_id: str) -> str:
    """Map imaging visit identifiers to standard PPMI EVENT_ID format.

    Args:
        visit_id: Raw visit identifier from XML metadata

    Returns:
        Standardized EVENT_ID (e.g., 'BL', 'V04', 'V06')

    Example:
        >>> map_visit_identifiers("baseline")
        'BL'
        >>> map_visit_identifiers("month_12")
        'V04'
    """
    if not visit_id:
        return "UNKNOWN"

    visit_lower = visit_id.lower().strip()

    # Common visit mapping patterns
    visit_mapping = {
        "baseline": "BL",
        "bl": "BL",
        "screening": "SC",
        "month_3": "V01",
        "month_6": "V02",
        "month_12": "V04",
        "month_18": "V05",
        "month_24": "V06",
        "month_36": "V08",
        "month_48": "V10",
        "year_1": "V04",
        "year_2": "V06",
        "year_3": "V08",
        "year_4": "V10",
        "v01": "V01",
        "v02": "V02",
        "v04": "V04",
        "v05": "V05",
        "v06": "V06",
        "v08": "V08",
        "v10": "V10",
    }

    # Try direct mapping first
    if visit_lower in visit_mapping:
        return visit_mapping[visit_lower]

    # Try pattern matching for numeric months
    if "month" in visit_lower:
        try:
            month_num = int("".join(filter(str.isdigit, visit_lower)))
            if month_num == 3:
                return "V01"
            if month_num == 6:
                return "V02"
            if month_num == 12:
                return "V04"
            if month_num == 18:
                return "V05"
            if month_num == 24:
                return "V06"
            if month_num == 36:
                return "V08"
            if month_num == 48:
                return "V10"
        except ValueError:
            pass

    logger.warning(f"Could not map visit identifier: {visit_id}, using raw value")
    return visit_id.upper()


def load_all_xml_metadata(
    xml_directory: str | Path, pattern: str = "*.xml"
) -> pd.DataFrame:
    """Load and parse all XML metadata files from a directory.

    Args:
        xml_directory: Path to directory containing XML files
        pattern: File pattern to match (default: "*.xml")

    Returns:
        DataFrame containing all parsed metadata with standardized columns

    Example:
        >>> df = load_all_xml_metadata("/path/to/xml/files/")
        >>> print(df.columns.tolist())
        ['PATNO', 'EVENT_ID', 'modality', 'dateAcquired', ...]
    """
    xml_dir = Path(xml_directory)

    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    # Find all XML files
    xml_files = list(xml_dir.glob(pattern))

    if not xml_files:
        logger.warning(f"No XML files found in {xml_dir} with pattern {pattern}")
        return pd.DataFrame()

    logger.info(f"Found {len(xml_files)} XML files to process")

    # Parse all XML files
    metadata_list = []
    successful_parses = 0

    for xml_file in xml_files:
        metadata = parse_xml_metadata(xml_file)
        if metadata is not None:
            metadata_list.append(metadata)
            successful_parses += 1
        else:
            logger.warning(f"Failed to parse XML file: {xml_file}")

    logger.info(f"Successfully parsed {successful_parses}/{len(xml_files)} XML files")

    if not metadata_list:
        logger.error("No XML files were successfully parsed")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(metadata_list)

    # Standardize column names for integration with PPMI data
    column_mapping = {"subjectIdentifier": "PATNO", "visitIdentifier": "EVENT_ID_RAW"}

    df = df.rename(columns=column_mapping)

    # Map visit identifiers to standard EVENT_ID format
    if "EVENT_ID_RAW" in df.columns:
        df["EVENT_ID"] = df["EVENT_ID_RAW"].apply(map_visit_identifiers)

    # Ensure PATNO is string type for consistent merging
    if "PATNO" in df.columns:
        df["PATNO"] = df["PATNO"].astype(str)

    # Add metadata about the loading process
    df["xml_parse_timestamp"] = datetime.now().isoformat()
    df["xml_source_directory"] = str(xml_dir)

    logger.info(f"Created imaging metadata DataFrame with shape {df.shape}")
    logger.info(
        f"Unique subjects: {df['PATNO'].nunique() if 'PATNO' in df.columns else 0}"
    )
    logger.info(
        f"Unique visits: {df['EVENT_ID'].nunique() if 'EVENT_ID' in df.columns else 0}"
    )

    return df


def validate_imaging_metadata(df: pd.DataFrame) -> dict[str, any]:
    """Validate the loaded imaging metadata DataFrame.

    Args:
        df: DataFrame containing imaging metadata

    Returns:
        Dictionary containing validation results and statistics
    """
    validation_results = {
        "total_records": len(df),
        "unique_subjects": df["PATNO"].nunique() if "PATNO" in df.columns else 0,
        "unique_visits": df["EVENT_ID"].nunique() if "EVENT_ID" in df.columns else 0,
        "missing_patno": df["PATNO"].isnull().sum() if "PATNO" in df.columns else 0,
        "missing_event_id": df["EVENT_ID"].isnull().sum()
        if "EVENT_ID" in df.columns
        else 0,
        "modalities": df["modality"].value_counts().to_dict()
        if "modality" in df.columns
        else {},
        "manufacturers": df["manufacturer"].value_counts().to_dict()
        if "manufacturer" in df.columns
        else {},
        "validation_passed": True,
        "issues": [],
    }

    # Check for critical missing values
    if validation_results["missing_patno"] > 0:
        validation_results["issues"].append(
            f"Missing PATNO in {validation_results['missing_patno']} records"
        )
        validation_results["validation_passed"] = False

    if validation_results["missing_event_id"] > 0:
        validation_results["issues"].append(
            f"Missing EVENT_ID in {validation_results['missing_event_id']} records"
        )
        validation_results["validation_passed"] = False

    # Log validation results
    if validation_results["validation_passed"]:
        logger.info("Imaging metadata validation passed")
    else:
        logger.warning(
            f"Imaging metadata validation failed: {validation_results['issues']}"
        )

    return validation_results


def normalize_modality(modality_str: str) -> str:
    """Standardize modality names from PPMI directory structure.

    Args:
        modality_str: Raw modality string from directory name

    Returns:
        Standardized modality name

    Example:
        >>> normalize_modality('DaTscan')
        'DATSCAN'
        >>> normalize_modality('SAG_3D_MPRAGE')
        'MPRAGE'
    """
    modality_lower = modality_str.lower().strip()

    # Handle MPRAGE variations
    if "mprage" in modality_lower:
        return "MPRAGE"

    # Handle DaTSCAN variations
    if "dat" in modality_lower and "scan" in modality_lower:
        return "DATSCAN"
    if "datscan" in modality_lower:
        return "DATSCAN"

    # Handle other common modalities
    if "dti" in modality_lower:
        return "DTI"
    if "flair" in modality_lower:
        return "FLAIR"
    if "swi" in modality_lower:
        return "SWI"
    if "bold" in modality_lower or "rest" in modality_lower:
        return "REST"

    # Default: return uppercase
    return modality_str.upper()


def create_ppmi_imaging_manifest(
    root_dir: str | Path, save_path: str | Path | None = None
) -> pd.DataFrame:
    """Scan PPMI directory structure to create comprehensive imaging manifest.

    Expected PPMI directory structure:
    root_dir/
    ├── {PATNO}/
    │   └── {MODALITY}/
    │       └── {TIMESTAMP}/
    │           └── I{SERIES_ID}/  # Contains DICOM files

    Args:
        root_dir: Path to PPMI root directory (e.g., "PPMI 2/")
        save_path: Optional path to save CSV manifest

    Returns:
        DataFrame with columns: PATNO, Modality, AcquisitionDate, SeriesUID, DicomPath

    Example:
        >>> manifest = create_ppmi_imaging_manifest("data/PPMI 2/")
        >>> print(f"Found {len(manifest)} imaging series")
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"PPMI root directory not found: {root_dir}")

    scan_metadata_list: list[dict] = []

    logger.info(f"Scanning PPMI directory: {root_path}")
    logger.info("This may take several minutes for large datasets...")

    # Use glob to find all potential series directories
    # Pattern: {PATNO}/{MODALITY}/{TIMESTAMP}/{SERIES_UID}
    try:
        series_patterns = [
            "*/*/*/*",  # Standard 4-level structure
            "*/*/*/*/*",  # Some datasets may have deeper nesting
        ]

        all_series_paths = []
        for pattern in series_patterns:
            paths = list(root_path.glob(pattern))
            all_series_paths.extend([p for p in paths if p.is_dir()])

        logger.info(f"Found {len(all_series_paths)} potential series directories")

        processed_count = 0
        for series_path in all_series_paths:
            try:
                # Only process directories that start with 'I' (DICOM series identifier)
                if not series_path.name.startswith("I"):
                    continue

                # Parse path structure relative to root
                parts = series_path.relative_to(root_path).parts

                # Need at least 4 parts: PATNO/MODALITY/TIMESTAMP/SERIES_ID
                if len(parts) < 4:
                    continue

                patno_str = parts[0]
                modality_raw = parts[1]
                timestamp_str = parts[2]
                series_uid = parts[3]

                # Validate PATNO (should be numeric)
                try:
                    patno = int(patno_str)
                except ValueError:
                    # Skip non-numeric patient IDs (likely phantom or test data)
                    if not any(char.isdigit() for char in patno_str):
                        continue
                    patno = patno_str  # Keep as string for mixed IDs

                # Extract acquisition date from timestamp
                # Format is typically: YYYY-MM-DD_HH_MM_SS.S
                try:
                    acquisition_date = timestamp_str.split("_")[0]
                    # Validate date format
                    datetime.strptime(acquisition_date, "%Y-%m-%d")
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse timestamp: {timestamp_str}")
                    acquisition_date = timestamp_str

                # Check if directory contains DICOM files
                dicom_files = list(series_path.glob("*.dcm")) + list(
                    series_path.glob("*.DCM")
                )
                if not dicom_files:
                    continue  # Skip empty directories

                scan_metadata_list.append(
                    {
                        "PATNO": patno,
                        "Modality": normalize_modality(modality_raw),
                        "ModalityRaw": modality_raw,  # Keep original for reference
                        "AcquisitionDate": acquisition_date,
                        "Timestamp": timestamp_str,
                        "SeriesUID": series_uid,
                        "DicomPath": str(series_path.resolve()),  # Absolute path
                        "DicomFileCount": len(dicom_files),
                    }
                )

                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} series...")

            except (IndexError, ValueError) as e:
                logger.debug(f"Could not parse path: {series_path}. Error: {e}")
                continue

    except Exception as e:
        logger.error(f"Error scanning directory structure: {e}")
        raise

    if not scan_metadata_list:
        logger.warning("No valid DICOM series found in directory structure")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(scan_metadata_list)
    logger.info(f"Successfully created manifest with {len(df)} imaging series")

    # Convert acquisition date to datetime
    try:
        df["AcquisitionDate"] = pd.to_datetime(df["AcquisitionDate"], format="%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Could not convert all dates to datetime: {e}")
        df["AcquisitionDate"] = pd.to_datetime(df["AcquisitionDate"], errors="coerce")

    # Sort by patient and acquisition date
    df = df.sort_values(["PATNO", "AcquisitionDate"], na_position="last").reset_index(
        drop=True
    )

    # Add summary statistics
    logger.info("Manifest summary:")
    logger.info(f"  - Unique patients: {df['PATNO'].nunique()}")
    logger.info(f"  - Modalities found: {df['Modality'].value_counts().to_dict()}")
    logger.info(
        f"  - Date range: {df['AcquisitionDate'].min()} to {df['AcquisitionDate'].max()}"
    )

    # Save manifest if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Manifest saved to: {save_path}")

    return df


def align_imaging_with_visits(
    imaging_manifest: pd.DataFrame,
    visit_data: pd.DataFrame,
    tolerance_days: int = 45,
    patno_col: str = "PATNO",
    visit_date_col: str = "INFODT",
    event_id_col: str = "EVENT_ID",
) -> pd.DataFrame:
    """Align imaging acquisition dates with PPMI visit dates to assign EVENT_IDs.

    Args:
        imaging_manifest: DataFrame from create_ppmi_imaging_manifest
        visit_data: DataFrame with visit information (PATNO, EVENT_ID, INFODT)
        tolerance_days: Maximum days between scan and visit date
        patno_col: Column name for patient ID in visit_data
        visit_date_col: Column name for visit date in visit_data
        event_id_col: Column name for event ID in visit_data

    Returns:
        Enhanced imaging manifest with EVENT_ID assignments

    Example:
        >>> aligned = align_imaging_with_visits(
        ...     imaging_manifest=manifest_df,
        ...     visit_data=ppmi_info_df
        ... )
    """
    if imaging_manifest.empty:
        logger.warning("Empty imaging manifest provided")
        return imaging_manifest

    if visit_data.empty:
        logger.warning("Empty visit data provided")
        return imaging_manifest

    # Prepare visit data
    visit_df = visit_data.copy()

    # Convert visit date to datetime
    visit_df[visit_date_col] = pd.to_datetime(visit_df[visit_date_col], errors="coerce")

    # Remove rows with invalid dates
    visit_df = visit_df.dropna(subset=[visit_date_col])

    # Initialize result columns
    imaging_aligned = imaging_manifest.copy()
    imaging_aligned["EVENT_ID"] = None
    imaging_aligned["MatchedVisitDate"] = None
    imaging_aligned["DaysDifference"] = None
    imaging_aligned["MatchQuality"] = None

    matched_count = 0

    logger.info(f"Aligning {len(imaging_manifest)} scans with visit data...")

    for idx, scan_row in imaging_manifest.iterrows():
        patno = scan_row["PATNO"]
        scan_date = scan_row["AcquisitionDate"]

        if pd.isna(scan_date):
            continue

        # Find visits for this patient
        patient_visits = visit_df[visit_df[patno_col] == patno].copy()

        if patient_visits.empty:
            continue

        # Calculate days difference between scan and each visit
        patient_visits["days_diff"] = abs(
            (patient_visits[visit_date_col] - scan_date).dt.days
        )

        # Find closest visit within tolerance
        within_tolerance = patient_visits[patient_visits["days_diff"] <= tolerance_days]

        if not within_tolerance.empty:
            # Get the closest match
            closest_match = within_tolerance.loc[within_tolerance["days_diff"].idxmin()]

            imaging_aligned.loc[idx, "EVENT_ID"] = closest_match[event_id_col]
            imaging_aligned.loc[idx, "MatchedVisitDate"] = closest_match[visit_date_col]
            imaging_aligned.loc[idx, "DaysDifference"] = closest_match["days_diff"]

            # Assign match quality
            if closest_match["days_diff"] <= 7:
                imaging_aligned.loc[idx, "MatchQuality"] = "excellent"
            elif closest_match["days_diff"] <= 21:
                imaging_aligned.loc[idx, "MatchQuality"] = "good"
            else:
                imaging_aligned.loc[idx, "MatchQuality"] = "acceptable"

            matched_count += 1

    match_rate = (matched_count / len(imaging_manifest)) * 100
    logger.info(
        f"Successfully matched {matched_count}/{len(imaging_manifest)} scans ({match_rate:.1f}%)"
    )

    # Summary statistics
    if matched_count > 0:
        quality_counts = imaging_aligned["MatchQuality"].value_counts()
        logger.info(f"Match quality distribution: {quality_counts.to_dict()}")

        avg_days_diff = imaging_aligned["DaysDifference"].mean()
        logger.info(f"Average days difference: {avg_days_diff:.1f}")

    return imaging_aligned


# Expose key functions
__all__ = [
    "parse_xml_metadata",
    "load_all_xml_metadata",
    "map_visit_identifiers",
    "validate_imaging_metadata",
    "normalize_modality",
    "create_ppmi_imaging_manifest",
    "align_imaging_with_visits",
]
