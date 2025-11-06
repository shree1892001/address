# Postman Testing Guide for OCR Extract API

## 🚀 API Server Information

- **Base URL**: `http://localhost:8000`
- **API Version**: `v1`
- **Full Base URL**: `http://localhost:8000/api/v1`

## 📋 Available Endpoints

### 1. **Health Check**
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/health`
- **Description**: Check if the API is running
- **Expected Response**:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00.000000",
    "service": "OCR Document Processing Service"
}
```

### 2. **Upload Interface**
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/`
- **Description**: Get the HTML upload interface
- **Expected Response**: HTML page

### 3. **Extract Table from PDF**
- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/extract-table`
- **Description**: Upload a PDF file and extract table data
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - Key: `file`
  - Type: `File`
  - Value: Select a PDF file
- **Expected Response**:
```json
{
    "success": true,
    "message": "Successfully extracted table from filename.pdf",
    "rows_extracted": 25,
    "columns": 5,
    "excel_file": "temp_uploads/uuid.xlsx",
    "timestamp": "2024-01-15T10:30:00.000000"
}
```

### 4. **Download Extracted CSV**
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/download/{file_id}`
- **Description**: Download the extracted table as CSV
- **Parameters**: Replace `{file_id}` with the UUID from the extract response
- **Expected Response**: CSV file download

### 5. **Test Download**
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/test-download`
- **Description**: Test file download functionality
- **Expected Response**: CSV file download

### 6. **Simple Table Extraction**
- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/extract-simple`
- **Description**: Simplified table extraction with data in response
- **Content-Type**: `multipart/form-data`
- **Body**:
  - Key: `file`
  - Type: `File`
  - Value: Select a PDF file
- **Expected Response**:
```json
{
    "success": true,
    "message": "Successfully extracted table from filename.pdf",
    "data": [
        ["Header1", "Header2", "Header3"],
        ["Data1", "Data2", "Data3"],
        ["Data4", "Data5", "Data6"]
    ],
    "rows": 2,
    "columns": 3
}
```

## 🛠️ Setting Up Postman

### 1. **Create a New Collection**
1. Open Postman
2. Click "New" → "Collection"
3. Name it "OCR Extract API"
4. Add description: "API for extracting tables from PDF documents"

### 2. **Set Up Environment Variables**
1. Click "Environments" → "New Environment"
2. Name it "OCR Extract Local"
3. Add variables:
   - `base_url`: `http://localhost:8000`
   - `api_version`: `v1`
   - `full_url`: `{{base_url}}/api/{{api_version}}`

### 3. **Create Request Templates**

#### Health Check Request
```
Method: GET
URL: {{full_url}}/health
```

#### Extract Table Request
```
Method: POST
URL: {{full_url}}/extract-table
Headers:
  - Content-Type: multipart/form-data
Body:
  - Key: file
  - Type: File
  - Value: [Select PDF file]
```

#### Download CSV Request
```
Method: GET
URL: {{full_url}}/download/{{file_id}}
```

## 📝 Step-by-Step Testing

### Step 1: Test Health Check
1. Create a new request in Postman
2. Set method to `GET`
3. Set URL to `http://localhost:8000/api/v1/health`
4. Click "Send"
5. **Expected**: 200 OK with health status

### Step 2: Test File Upload
1. Create a new request in Postman
2. Set method to `POST`
3. Set URL to `http://localhost:8000/api/v1/extract-table`
4. Go to "Body" tab
5. Select "form-data"
6. Add key: `file`, type: `File`
7. Select a PDF file with tables
8. Click "Send"
9. **Expected**: 200 OK with extraction results

### Step 3: Test Download
1. Copy the `file_id` from the previous response
2. Create a new request
3. Set method to `GET`
4. Set URL to `http://localhost:8000/api/v1/download/{file_id}`
5. Click "Send"
6. **Expected**: File download

## 🔍 Testing Different Scenarios

### 1. **Valid PDF with Tables**
- Use a PDF file that contains clear table data
- Expected: Successful extraction with table data

### 2. **Invalid File Type**
- Try uploading a non-PDF file (e.g., .txt, .docx)
- Expected: 400 Bad Request with validation error

### 3. **PDF without Tables**
- Use a PDF file that doesn't contain tables
- Expected: 422 Unprocessable Entity with "No table found" error

### 4. **Corrupted PDF**
- Use a corrupted or invalid PDF file
- Expected: 422 Unprocessable Entity with PDF processing error

### 5. **Large PDF File**
- Test with a large PDF file
- Expected: Processing with performance monitoring

## 📊 Response Status Codes

| Status Code | Meaning | Example |
|-------------|---------|---------|
| 200 | Success | Health check, successful extraction |
| 400 | Bad Request | Invalid file type, validation errors |
| 404 | Not Found | File not found, invalid file ID |
| 422 | Unprocessable Entity | PDF processing errors, no tables found |
| 500 | Internal Server Error | Server errors, configuration issues |

## 🔧 Error Handling Examples

### Validation Error (400)
```json
{
    "detail": "Unsupported file type: txt. Supported types: pdf"
}
```

### File Not Found (404)
```json
{
    "detail": "File not found: /path/to/file.pdf"
}
```

### No Table Found (422)
```json
{
    "detail": "No table found in document: /path/to/file.pdf"
}
```

### PDF Processing Error (422)
```json
{
    "detail": "PDF file is corrupted: /path/to/file.pdf"
}
```

## 🎯 Advanced Testing

### 1. **Performance Testing**
- Monitor the logs for performance metrics
- Check execution times in the response
- Test with different file sizes

### 2. **Retry Mechanism Testing**
- Test with files that might cause temporary failures
- Monitor retry attempts in logs
- Verify cleanup functions work

### 3. **AOP Exception Handling**
- Test different error scenarios
- Verify proper exception mapping
- Check severity-based logging

## 📁 Sample Test Files

### Recommended Test PDFs:
1. **Simple Table**: PDF with basic table structure
2. **Complex Table**: PDF with merged cells, formatting
3. **Multi-page Table**: PDF with tables spanning multiple pages
4. **Arabic Text**: PDF with Arabic text and tables
5. **Mixed Content**: PDF with tables and other content

## 🚨 Troubleshooting

### Common Issues:

1. **Server Not Starting**
   - Check if port 8000 is available
   - Verify all dependencies are installed
   - Check Python version compatibility

2. **Import Errors**
   - Run `python test_imports.py` to verify imports
   - Check if all required packages are installed

3. **File Upload Issues**
   - Ensure file is actually a PDF
   - Check file size limits
   - Verify file permissions

4. **Extraction Failures**
   - Check if PDF contains extractable tables
   - Verify PDF is not corrupted
   - Check OCR dependencies (Tesseract)

## 📞 Support

If you encounter issues:
1. Check the application logs in the console
2. Verify the API server is running on port 8000
3. Test with the health check endpoint first
4. Use the test download endpoint to verify basic functionality

## 🎉 Success Indicators

- Health check returns 200 OK
- File upload completes successfully
- Table data is extracted and returned
- CSV download works correctly
- Error responses are properly formatted
- Logs show AOP exception handling in action
