import docx
import PyPDF2
import csv
import os
from typing import List, Dict, Any, Tuple

class LocationParser:
    """
    Class to parse location data from various document formats.
    Supports CSV, TXT, PDF, and DOCX files.
    """
    
    def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse location data from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of location dictionaries with 'name', 'address', 'latitude', 'longitude'
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.csv':
            return self._parse_csv(file_path)
        elif file_extension == '.txt':
            return self._parse_txt(file_path)
        elif file_extension == '.pdf':
            return self._parse_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _parse_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse location data from CSV file."""
        locations = []
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                location = self._process_location_data(row)
                if location:
                    locations.append(location)
        return locations
    
    def _parse_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse location data from TXT file."""
        locations = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        # Assuming each location is on a separate line with comma or tab separators
        for line in lines:
            if ',' in line:
                parts = [part.strip() for part in line.split(',')]
            else:
                parts = [part.strip() for part in line.split('\t')]
            
            if len(parts) >= 2:  # At minimum, we need name and address
                location = {
                    'name': parts[0],
                    'address': parts[1],
                }
                
                # Add coordinates if available
                if len(parts) >= 4:
                    try:
                        location['latitude'] = float(parts[2])
                        location['longitude'] = float(parts[3])
                    except ValueError:
                        # If coordinates are not valid, we'll geocode later
                        pass
                
                locations.append(location)
        
        return locations
    
    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse location data from PDF file."""
        locations = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        # Process extracted text line by line
        lines = text.split('\n')
        for line in lines:
            if ',' in line and len(line.split(',')) >= 2:
                parts = [part.strip() for part in line.split(',')]
                location = self._process_location_parts(parts)
                if location:
                    locations.append(location)
        
        return locations
    
    def _parse_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse location data from DOCX file."""
        locations = []
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text and ',' in text:
                parts = [part.strip() for part in text.split(',')]
                location = self._process_location_parts(parts)
                if location:
                    locations.append(location)
        return locations
    
    def _process_location_data(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Process a location data dictionary to standardize keys and values."""
        location = {}
        
        # Map common column names to standard keys
        name_keys = ['name', 'location name', 'location', 'place']
        address_keys = ['address', 'location address', 'full address']
        lat_keys = ['latitude', 'lat', 'y']
        lng_keys = ['longitude', 'lng', 'long', 'x']
        
        # Find the name
        for key in name_keys:
            if key in data:
                location['name'] = data[key]
                break
        
        # Find the address
        for key in address_keys:
            if key in data:
                location['address'] = data[key]
                break
        
        # Find coordinates
        for key in lat_keys:
            if key in data:
                try:
                    location['latitude'] = float(data[key])
                except (ValueError, TypeError):
                    pass
        
        for key in lng_keys:
            if key in data:
                try:
                    location['longitude'] = float(data[key])
                except (ValueError, TypeError):
                    pass
        
        # Require at least a name or address
        if 'name' in location or 'address' in location:
            return location
        return None
    
    def _process_location_parts(self, parts: List[str]) -> Dict[str, Any]:
        """Process a list of strings that may contain location data."""
        if len(parts) < 2:
            return None
        
        location = {
            'name': parts[0],
            'address': parts[1]
        }
        
        # Try to extract coordinates if available
        if len(parts) >= 4:
            try:
                location['latitude'] = float(parts[2])
                location['longitude'] = float(parts[3])
            except ValueError:
                pass
        
        return location