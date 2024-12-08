import logging
from twilio.rest import Client

# Setup logging
logging.basicConfig(filename='tmp/model_performance.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to log performance metrics
def log_performance(accuracy, precision, recall, f1):
    logging.info(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

# Twilio setup
account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
twilio_phone_number = 'YOUR_TWILIO_PHONE_NUMBER'
recipient_phone_number = 'RECIPIENT_PHONE_NUMBER'

client = Client(account_sid, auth_token)

# Function to send SMS alerts
def send_sms_alert(message):
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )
# Graceful stop setup 
stop_monitoring = False 

def signal_handler(sig, frame): 
  global stop_monitoring print('Stopping monitoring...') 
  stop_monitoring = True
  # Register the signal handler 
signal.signal(signal.SIGINT, signal_handler)

# Periodic Evaluation with Alerts
import time

# Function to simulate continuous monitoring
def continuous_monitoring(model, X_val, y_val, interval=60):
    while True:
        y_val_pred = model.predict(X_val).round()
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)

        # Log performance metrics
        log_performance(val_accuracy, val_precision, val_recall, val_f1)

        # Print performance metrics to the console
        print(f'Validation Accuracy: {val_accuracy:.2f}')
        print(f'Validation Precision: {val_precision:.2f}')
        print(f'Validation Recall: {val_recall:.2f}')
        print(f'Validation F1 Score: {val_f1:.2f}')
        
        # Alert mechanism (example threshold)
        if val_accuracy < 0.80:
            alert_message = f"Alert: Model performance has dropped below threshold! Accuracy: {val_accuracy:.2f}"
            print(alert_message)
            send_sms_alert(alert_message)

        
        # Sleep for the interval duration
        time.sleep(interval)

# Simulate continuous monitoring # This will run indefinitely until stopped by a keyboard interrupt (Ctrl+C)
try: 
      continuous_monitoring(model, X_val, y_val, interval=60) 
except 
      KeyboardInterrupt: print("Monitoring stopped by user.")
