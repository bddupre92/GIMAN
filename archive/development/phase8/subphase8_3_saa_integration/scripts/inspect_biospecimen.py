"""Quick script to inspect biospecimen data for alpha-synuclein."""

import pandas as pd

# Load the biospecimen data
df = pd.read_csv('data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv')

print("=" * 80)
print("BIOSPECIMEN DATA STRUCTURE")
print("=" * 80)
print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumns: {', '.join(df.columns.tolist())}")

print("\n" + "=" * 80)
print("UNIQUE TEST NAMES (first 50)")
print("=" * 80)
test_names = df['TESTNAME'].unique()
for i, test in enumerate(test_names[:50], 1):
    print(f"{i:3d}. {test}")

print(f"\n... Total unique tests: {len(test_names)}")

# Search for alpha-synuclein related tests
print("\n" + "=" * 80)
print("ALPHA-SYNUCLEIN RELATED TESTS")
print("=" * 80)
alpha_mask = df['TESTNAME'].str.contains('alpha|syn', case=False, na=False)
alpha_tests = df[alpha_mask]['TESTNAME'].unique()

if len(alpha_tests) > 0:
    print(f"\nFound {len(alpha_tests)} alpha-synuclein related tests:")
    for test in alpha_tests:
        count = len(df[df['TESTNAME'] == test])
        print(f"  - {test}: {count:,} measurements")
        
    # Show sample data
    print("\n" + "=" * 80)
    print("SAMPLE ALPHA-SYNUCLEIN DATA")
    print("=" * 80)
    sample = df[alpha_mask].head(10)
    print(sample[['PATNO', 'CLINICAL_EVENT', 'TESTNAME', 'TESTVALUE', 'UNITS']].to_string())
else:
    print("\nNo alpha-synuclein related tests found.")
    print("\nSearching for CSF biomarkers...")
    csf_mask = df['TYPE'].str.contains('CSF', case=False, na=False)
    csf_tests = df[csf_mask]['TESTNAME'].unique()
    print(f"\nFound {len(csf_tests)} CSF-related tests:")
    for test in csf_tests[:20]:
        count = len(df[df['TESTNAME'] == test])
        print(f"  - {test}: {count:,} measurements")

print("\n" + "=" * 80)
print("DATA SUMMARY BY TYPE")
print("=" * 80)
print(df['TYPE'].value_counts())
