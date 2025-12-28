from vehicle.autonomous_vehicle import AutonomousVehicle
import carla
import time
import math
import cv2
import numpy as np
import queue
import sys

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
av = AutonomousVehicle(client)
world, map = av.spawn_world('Town01')
vehicle = av.spawn_vehicle('vehicle.mercedes.coupe')
av.equip_sensors()

# Prepare per-sensor queues
camera_queues = {}
lidar_queue = queue.Queue(maxsize=8)

def make_camera_callback(key):
    def _callback(image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            frame = arr[:, :, :3][:, :, ::-1]
            q = camera_queues.get(key)
            if q is not None and not q.full():
                q.put(frame)
        except Exception:
            pass
    return _callback

def lidar_callback(point_cloud):
    try:
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))  # x, y, z, intensity
        if not lidar_queue.full():
            lidar_queue.put(points)
    except Exception:
        pass

def render_lidar(points, img_size=800, scale=5.0):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    if points.size == 0:
        return img
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    # simple top-down projection: x forward, y left/right
    cx = img_size // 2
    cy = img_size // 2
    us = (xs * scale + cx).astype(np.int32)
    vs = (-ys * scale + cy).astype(np.int32)
    mask = (us >= 0) & (us < img_size) & (vs >= 0) & (vs < img_size)
    us = us[mask]
    vs = vs[mask]
    zs = zs[mask]
    # color by height (z)
    if zs.size:
        zmin = np.min(zs)
        zmax = np.max(zs)
        denom = (zmax - zmin) if (zmax - zmin) != 0 else 1.0
        cols = ((zs - zmin) / denom * 255).astype(np.uint8)
        for (u, v, c) in zip(us, vs, cols):
            img = cv2.circle(img, (u, v), 1, (int(c), 255 - int(c), 128), -1)
    return img


# Attach listeners for all sensors created in autonomous_vehicle
for name, sensor in av.sensors.items():
    if sensor is None:
        continue
    if 'lidar' in name:
        sensor.listen(lidar_callback)
    else:
        camera_queues[name] = queue.Queue(maxsize=4)
        sensor.listen(make_camera_callback(name))

# Position spectator behind and above vehicle
spectator = world.get_spectator()
vehicle_transform = vehicle.get_transform()
spectator_transform = carla.Transform(
    vehicle_transform.location + carla.Location(y=-10, z=5),
    carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
)
spectator.set_transform(spectator_transform)

distance_traveled = 0
target_distance = 10
initial_location = vehicle.get_location()
vehicle.set_autopilot(True)

try:
    while True:
        current_location = vehicle.get_location()
        distance_traveled = math.sqrt(
            (current_location.x - initial_location.x) ** 2 +
            (current_location.y - initial_location.y) ** 2
        )

        vehicle_transform = vehicle.get_transform()
        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(y=-10, z=5),
            carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
        )
        spectator.set_transform(spectator_transform)

        cam_order = [
            'front_narrow', 'front_wide', 'left_front', 'right_front',
            'left_side', 'right_side', 'rear'
        ]

        thumbs = []
        thumb_size = (360, 202)  

        # collect camera thumbnails
        for name in cam_order:
            q = camera_queues.get(name)
            label = name
            if q is not None and not q.empty():
                frame = q.get()
                thumb = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)
            else:
                thumb = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
                cv2.putText(thumb, f"{label} (no data)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            cv2.putText(thumb, label, (10, thumb.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            thumbs.append(thumb)

        # LiDAR thumbnail
        if not lidar_queue.empty():
            points = lidar_queue.get()
            lidar_img = render_lidar(points, img_size=thumb_size[1], scale=6.0)
            lidar_thumb = cv2.resize(lidar_img, thumb_size, interpolation=cv2.INTER_AREA)
        else:
            lidar_thumb = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
            cv2.putText(lidar_thumb, "LiDAR (no data)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)
        cv2.putText(lidar_thumb, 'lidar', (10, lidar_thumb.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        thumbs.append(lidar_thumb)

        # Make grid (2 rows x 4 cols)
        cols = 4
        rows = int(math.ceil(len(thumbs)/cols))
        row_imgs = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r*cols + c
                if idx < len(thumbs):
                    row_tiles.append(thumbs[idx])
                else:
                    # filler
                    row_tiles.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))
            row_imgs.append(cv2.hconcat(row_tiles))
        grid = cv2.vconcat(row_imgs)

        # Overlay HUD info
        cv2.rectangle(grid, (0,0), (300,40), (0,0,0), -1)
        cv2.putText(grid, f"Distance: {distance_traveled:.2f} m", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('Sensor Suite', grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('\nUser requested exit')
            break

        print(f"Distance traveled: {distance_traveled:.2f} meters", end='\r')
        time.sleep(0.05)

except KeyboardInterrupt:
    print('\nInterrupted by user')
