import sqlite3
import json
import argparse
from datetime import datetime

"""
MessageUtility.py
=================

A utility script for accessing and exporting data from the macOS Messages app SQLite database.

Description:
-----------
This script connects to the Messages app's chat.db SQLite database located in the user's Library
folder and provides functionality to list users (with their phone numbers/IDs) and export message
history for specific users. The output can be filtered by date ranges and message exports are 
saved to a log file.

Configuration Requirements:
--------------------------
The script requires a configuration file named 'messageUtility.json' in the same directory.
This file should contain a JSON object with the following structure:

    {
        "user": "your_username_here"
    }

Replace 'your_username_here' with your macOS username to point to the correct location of
the chat.db file (typically /Users/<username>/Library/Messages/chat.db).

Usage Examples:
--------------
1. List all users in the database:
   python3 messageUtility.py --listusers

2. List users with messages in a specific date range:
   python3 messageUtility.py --listusers --startdate 2023-01-01 --enddate 2023-12-31

3. Export all messages from a specific user to messageExport.log:
   python3 messageUtility.py --exportdata --fromuser +11234567890

4. Export messages from a specific user within a date range:
   python3 messageUtility.py --exportdata --fromuser user@example.com --startdate 2023-01-01 --enddate 2023-12-31

Notes:
-----
- The default operation (if no arguments are provided) is to list all users.
- Date formats must be specified as YYYY-MM-DD.
- User identifiers can be phone numbers (e.g., +11234567890) or email addresses for iMessage.
- Messages are exported to a file named 'messageExport.log' in the current directory.
- This script only reads from the database and does not modify any data.
- If the chat.db file is locked (Messages app is open), you may need to close the Messages app
  or make a copy of the database file to a different location.
"""

def load_config():
    try:
        with open('messageUtility.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Configuration file messageUtility.json not found!")
        exit(1)


def list_users(db_path, start_date=None, end_date=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT handle.id, handle.uncanonicalized_id FROM handle"
    if start_date and end_date:
        query += " JOIN message ON handle.ROWID = message.handle_id WHERE date(strftime('%Y-%m-%d %H:%M:%S', datetime(message.date/1000000000 + 978307200, 'unixepoch', 'localtime'))) BETWEEN ? AND ?"
        cursor.execute(query, (start_date, end_date))
    else:
        cursor.execute(query)

    users = cursor.fetchall()
    conn.close()
    return users


def export_messages(db_path, user, start_date=None, end_date=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT message.text, message.date
        FROM message
        JOIN handle ON handle.ROWID = message.handle_id
        WHERE handle.id = ?
    """
    params = [user]

    if start_date and end_date:
        query += " AND date(strftime('%Y-%m-%d %H:%M:%S', datetime(message.date/1000000000 + 978307200, 'unixepoch', 'localtime'))) BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    cursor.execute(query, params)
    messages = cursor.fetchall()
    conn.close()
    return messages


def main():
    config = load_config()
    db_path = f"/Users/{config['user']}/Library/Messages/chat.db"

    parser = argparse.ArgumentParser(description='Message Utility')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--listusers', action='store_true', help='List users')
    group.add_argument('--exportdata', action='store_true',
                       help='Export messages')
    parser.add_argument('--fromuser', help='User to export messages from')
    parser.add_argument('--startdate', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--enddate', help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.exportdata:
        if not args.fromuser:
            print("--fromuser is required for --exportdata")
            exit(1)
        messages = export_messages(
            db_path, args.fromuser, args.startdate, args.enddate)
        
        # Write messages to messageExport.log instead of printing to console
        with open("messageExport.log", "w") as log_file:
            for message, timestamp in messages:
                formatted_message = f"[{datetime.fromtimestamp(timestamp/1000000000 + 978307200)}] {message}"
                log_file.write(formatted_message + "\n")
        
        print(f"Messages exported to messageExport.log")
    else:
        users = list_users(db_path, args.startdate, args.enddate)
        for user_id, phone in users:
            print(f"User: {user_id}, Phone: {phone}")


if __name__ == "__main__":
    main()
