# IoT Forecast Dashboard

Interactive Streamlit dashboard to search and view Excel data and upload documents linked to specific rows.

## Features

- Upload or select an Excel file (`.xlsx`) stored under `data/`
- Global search across all columns
- Column-aware filters (categorical multi-select, numeric range, date range)
- Interactive table view with CSV export of filtered data
- Select a key column and attach documents to rows; files stored under `uploads/<row_key>/`
- List and download existing attachments for a selected row

## Quick Start

### 1) Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
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
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── uploads/

## Contact

For questions or support, contact: sharkh.m91@gmail.com
```
