import asyncio
from bleak import BleakScanner, BleakClient

# Important address from Bluefruit chip
target_mac_address = "F4:7D:A4:2C:1E:EE" 
SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
TX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
RX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

#scans for the device
async def scan_for_device():
    print("Scanning for nearby BLE devices...")
    devices = await BleakScanner.discover()
    for device in devices:
        if device.address == target_mac_address:
            print("Found target device:", device.name)
            return device
    print("Target device not found.")
    return None

#connects to device
async def connect_to_device(device):
    if device is None:
        return None
    print("Connecting to target device...")
    client = BleakClient(device)
    try:
        await client.connect()
        print("Connected to target device successfully!")
        return client
    except Exception as e:
        print("Failed to connect to target device:", e)
        return None

async def main():
    target_device = await scan_for_device()
    if target_device:
        client = await connect_to_device(target_device)
        #THIS IS THE CODE THAT WILL RUN ONCE CONNECTED!
        if client:
            while True:
                # Receive data
                received_data = await client.read_gatt_char(RX_CHARACTERISTIC_UUID)
                print("Received data:", received_data)
                await asyncio.sleep(1)  # Adjust sleep time as needed
        else:
            print("Failed to connect to target device.")
    else:
        print("Exiting.")

if __name__ == "__main__":
    asyncio.run(main())
