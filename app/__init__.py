from flask import Flask
from app.api import api  # Import the blueprint

app = Flask(__name__)

# Register the blueprint with the app
app.register_blueprint(api)

# Add any other initialization code here if needed
