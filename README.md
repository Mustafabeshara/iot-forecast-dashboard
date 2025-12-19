# IoT Forecast Dashboard

Interactive Streamlit dashboard to search and view Excel data and upload documents linked to specific rows.

## Features

- Upload or select an Excel file (`.xlsx`) stored under `data/`
- Global search across all columns
- Column-aware filters (categorical multi-select, numeric range, date range)
- Interactive table view with CSV export of filtered data
- Select a key column and attach documents to rows; files stored under `uploads/<row_key>/`
- List and download existing attachments for a selected row
- **Comprehensive tracing and logging** for monitoring application behavior and debugging

## Tracing & Logging

The application includes a comprehensive tracing system that logs:

- **User Actions**: File uploads, searches, filters, downloads, attachments
- **Data Operations**: Load, filter, search operations with row/column counts
- **Performance Metrics**: Execution time for key operations
- **Error Tracking**: Detailed exception logging with stack traces
- **Application Flow**: Function entry/exit points

### Log Configuration

Logs are written to:
- **Console**: Real-time logging output (INFO level and above)
- **Log File**: `logs/app.log` (created automatically, includes all log levels)

### Viewing Logs

```bash
# View real-time logs while running the app
tail -f logs/app.log

# Search for specific operations
grep "User action" logs/app.log
grep "Data operation" logs/app.log
grep "ERROR" logs/app.log
```

### Log Levels

The application uses standard Python logging levels:
- `INFO`: Normal operations, user actions, data operations
- `DEBUG`: Detailed diagnostic information
- `ERROR`: Error conditions with stack traces
- `WARNING`: Warning messages for potential issues

## Quick Start

### 1) Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
pip install -r dev-requirements.txt  # for tests
```

### 3) Run the app

```bash
streamlit run app.py
```

Then open the provided local URL (typically http://localhost:8501).

## Usage Notes

- Place your Excel files in the `data/` folder or upload via the sidebar.
- Choose a "Key Column" that uniquely identifies rows (e.g., `ID` or `Name`).
- When you select a row key value, you can upload documents that will be stored in `uploads/<row_key>/`.
- Use the "Download filtered CSV" button to export current filtered results.

## Project Structure

```
Forecast-dashboard/
├── app.py                  # Main Streamlit application
├── tracing.py              # Tracing and logging utilities
├── requirements.txt        # Production dependencies
├── dev-requirements.txt    # Development dependencies
├── README.md               # This file
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD pipeline
├── data/                   # Excel files storage
├── uploads/                # User attachments storage
├── logs/                   # Application logs
└── tests/                  # Unit tests
    ├── test_app.py         # Tests for main application
    └── test_tracing.py     # Tests for tracing module
```

## Tests

Run unit tests locally:

```bash
pytest -v
```

Or run specific test files:

```bash
pytest tests/test_app.py -v
pytest tests/test_tracing.py -v
```

## Contact

For questions or support, contact: sharkh.m91@gmail.com
