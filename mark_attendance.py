import pandas as pd
from datetime import datetime

# Function to mark attendance in a CSV file
def markAttendance(name):
    filename = 'Attendance.csv'
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Time'])

    # Check if the name is already marked
    if name not in df['Name'].values:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        df = pd.concat((df,pd.DataFrame([{'Name': name, 'Time': dtString}])), ignore_index=True)
        df.to_csv(filename, index=False)

# Example usage:
# markAttendance("John Doe")