# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Set environment variables for your credentials
# Read more at http://twil.io/secure

account_sid = ""
auth_token = ""
client = Client(account_sid, auth_token)

call = client.calls.create(
  url="callback_url",
  to="",
  from_=""
)

# print(call.sid)