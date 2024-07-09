import sys
import os

# Add the directory containing your application code to Python's path
sys.path.insert(0, '/var/www/betterbus-backend/code')

# Replace 'your_application' with the actual name of your Flask app instance
from detect import app as application

# Optional: Adjust application settings as needed
application.debug = True
