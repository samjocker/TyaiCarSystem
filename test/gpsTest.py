import geocoder

def is_within_rectangle(latitude, longitude, rect_coordinates):
    # rect_coordinates 是一個包含四組座標的列表，每組座標是一個包含緯度和經度的元組 (latitude, longitude)
    # 首先判斷經緯度是否在矩形範圍內
    if (
        rect_coordinates[0][0] <= latitude <= rect_coordinates[2][0] and
        rect_coordinates[0][1] <= longitude <= rect_coordinates[2][1]
    ):
        return True
    else:
        return False

def get_current_gps_coordinates():
    g = geocoder.ip('me')#this function is used to find the current information using our IP Add
    if g.latlng is not None: #g.latlng tells if the coordiates are found or not
        return g.latlng
    else:
        return None

if __name__ == "__main__":
    coordinates = get_current_gps_coordinates()
    rect_coordinates = [(24.99216, 121.32138), (24.99229, 121.32170), (24.99158, 121.32129), (24.99185, 121.32214)]
    if coordinates is not None:
        latitude, longitude = coordinates
        print(f"Your current GPS coordinates are:")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        if is_within_rectangle(latitude, longitude, rect_coordinates):
            print("in")
        else:
            print("out")
    else:
        print("Unable to retrieve your GPS coordinates.")