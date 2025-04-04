import cv2
import smtplib
from email.mime.text import MIMEText

def putTextRect(img, text, pos, scale=1, thickness=2, colorR=(0, 0, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colorR, thickness)

def send_email(to_email, subject, body):
    sender_email = "your_email@gmail.com"
    sender_password = "your_password"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("✅ Alert email sent!")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False
