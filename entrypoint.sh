#!/bin/bash
set -e

MODEL_PATH="/app/models/face_detection.pth"
GDRIVE_FILE_ID="1ZS7S05jn04c9TUVDIZfl9r0AqhiXgWle"

if [ ! -f "$MODEL_PATH" ]; then
    echo "=============================================="
    echo "  Modelo no encontrado en $MODEL_PATH"
    echo "  Descargando desde Google Drive..."
    echo "=============================================="
    mkdir -p /app/models
    python -c "
import urllib.request, re, sys

file_id = '$GDRIVE_FILE_ID'
output = '$MODEL_PATH'

# First request to get confirmation token
url = f'https://drive.google.com/uc?export=download&id={file_id}'
req = urllib.request.Request(url)
req.add_header('User-Agent', 'Mozilla/5.0')

try:
    resp = urllib.request.urlopen(req)
    # Check if we got a confirmation page (large file)
    content_type = resp.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        html = resp.read().decode()
        # Extract confirm token
        match = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
        if match:
            token = match.group(1)
            url = f'https://drive.google.com/uc?export=download&confirm={token}&id={file_id}'
        else:
            # Try uuid approach
            match = re.search(r'name=\"uuid\" value=\"([^\"]+)\"', html)
            if match:
                uuid = match.group(1)
                url = f'https://drive.google.com/uc?export=download&id={file_id}&uuid={uuid}&confirm=t'

    print(f'Descargando modelo...')
    urllib.request.urlretrieve(url, output)
    
    import os
    size_mb = os.path.getsize(output) / (1024*1024)
    print(f'Modelo descargado: {size_mb:.1f} MB')
except Exception as e:
    print(f'Error descargando modelo: {e}')
    print(f'Por favor descargue manualmente desde:')
    print(f'https://drive.google.com/file/d/{file_id}/view?usp=sharing')
    print(f'Y coloque el archivo en models/face_detection.pth')
"
    echo "=============================================="
else
    echo "Modelo encontrado en $MODEL_PATH"
fi

# Execute the original command
exec "$@"
