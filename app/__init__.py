from flask import Flask

app = Flask(__name__)

# Import routes
from app.api import enhance  # Ensure routes are registered
