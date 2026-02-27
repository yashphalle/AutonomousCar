from vehicle.autonomous_vehicle import AutonomousVehicle
import carla
import time
import math

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
av = AutonomousVehicle(client)
world,map = av.spawn_world('Town01')
vehicle = av.spawn_vehicle('vehicle.mercedes.coupe')
av.equip_sensors()

spectator = world.get_spectator()
vehicle_transform = vehicle.get_transform()
spectator_transform = carla.Transform (
    vehicle_transform.location + carla.Location(y= -10, z=5),
    carla.Rotation(pitch=-15,yaw = vehicle_transform.rotation.yaw)
)
spectator.set_transform(spectator_transform)

distance_traveled = 0
target_distance = 10
initial_location = vehicle.get_location()
vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

while(distance_traveled<target_distance):
    current_location = vehicle.get_location()

    distance_traveled = math.sqrt(
        (current_location.x - initial_location.x)**2+
        (current_location.y - initial_location.y)**2
    )
    vehicle_transform = vehicle.get_transform()
    spectator_transform = carla.Transform (
    vehicle_transform.location + carla.Location(y= -10, z=5),
    carla.Rotation(pitch=-15,yaw = vehicle_transform.rotation.yaw)
    )
    spectator.set_transform(spectator_transform)
    print(f"Distance traveled: {distance_traveled:.2f} meters", end='\r')
    time.sleep(0.05)
# Stop the vehicle
vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
print(f"\nTarget reached! Final distance: {distance_traveled:.2f} meters")
        
        # Final position
final_location = vehicle.get_location()
print(f"Final position: x={final_location.x:.2f}, y={final_location.y:.2f}, z={final_location.z:.2f}")

