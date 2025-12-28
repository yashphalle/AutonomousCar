import carla
import time
class AutonomousVehicle:
    def __init__(self,client):
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.vehicle = None
        self.sensor_data = {}
        self.town_name = client.get_world().get_map().name
        self.blueprint_library = self.world.get_blueprint_library()
        self.sensors = {}
    
    def spawn_world(self,town_name):
        self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()
        return self.world,self.map


    def spawn_vehicle(self,vehicle_model):
        blueprint = self.world.get_blueprint_library().find(vehicle_model)
        if not blueprint:
            raise ValueError(f"Vehicle model {vehicle_model} not found")
        spawn_point = self.world.get_map().get_spawn_points()[0]
        vehicle = self.world.spawn_actor(blueprint,spawn_point)
        self.vehicle = vehicle
        return vehicle
    
    def equip_sensors(self):
        """
        Equip the vehicle with 7 cameras + 1 LiDAR sensor.
        
        Camera Configuration:
        - 2 Front cameras (narrow 50° + wide 120°) - same position, different FOV
        - 2 Front diagonal cameras (left/right 60°)
        - 2 Side cameras (left/right 90°)
        - 1 Rear camera (120°)
        - 1 Roof-mounted 360° LiDAR
        """
        print("\nEquipping sensor suite...")
        
        self.sensors['front_narrow'] = self._add_camera(
            x=2.5, y=0.0, z=1.0, pitch=-10, yaw=0, roll=0,
            fov=50, image_x=1920, image_y=1080,
            name="Front Narrow (50° - Traffic Lights)"
        )
        
        self.sensors['front_wide'] = self._add_camera(
            x=2.5, y=0.0, z=1.0, pitch=-10, yaw=0, roll=0,
            fov=120, image_x=1920, image_y=1080,
            name="Front Wide (120° - Close Range)"
        )

        self.sensors['left_front'] = self._add_camera(
            x=2.0, y=-0.8, z=1.2, pitch=0, yaw=-45, roll=0,
            fov=60, image_x=1920, image_y=1080,
            name="Left Front Diagonal (60°)"
        )
        
        self.sensors['right_front'] = self._add_camera(
            x=2.0, y=0.8, z=1.2, pitch=0, yaw=45, roll=0,
            fov=60, image_x=1920, image_y=1080,
            name="Right Front Diagonal (60°)"
        )
        
        self.sensors['left_side'] = self._add_camera(
            x=0.0, y=-1.0, z=1.2, pitch=0, yaw=-90, roll=0,
            fov=90, image_x=1920, image_y=1080,
            name="Left Side (90°)"
        )
        
        self.sensors['right_side'] = self._add_camera(
            x=0.0, y=1.0, z=1.2, pitch=0, yaw=90, roll=0,
            fov=90, image_x=1920, image_y=1080,
            name="Right Side (90°)"
        )
        
        # === REAR CAMERA ===
        self.sensors['rear'] = self._add_camera(
            x=-2.0, y=0.0, z=1.2, pitch=-10, yaw=180, roll=0,
            fov=120, image_x=1920, image_y=1080,
            name="Rear Wide (120°)"
        )

        self.sensors['lidar'] = self._add_lidar(
            x=0.0, y=0.0, z=1.8,
            name="Roof LiDAR 360°"
        )
        
        print(f"\n✓ Sensor suite equipped: {len(self.sensors)} sensors")
        print(f"  - 7 RGB Cameras (1920x1080)")
        print(f"  - 1 LiDAR (360°, 32 channels, 100m range)")
        
        return self.sensors
    
    def _add_camera(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0,
                    fov=110, image_x=1920, image_y=1080, name="Camera"):
        """
        Internal method to add a camera sensor.
        """
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_x))
        camera_bp.set_attribute('image_size_y', str(image_y))
        camera_bp.set_attribute('fov', str(fov))
        
        camera_transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        )
        
        camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicle
        )
        
        print(f"  ✓ {name}")
        
        
        return camera
    
    def _add_lidar(self, x=0, y=0, z=0, name="LiDAR"):
        """
        Internal method to add a LiDAR sensor.
        """
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('points_per_second', '1280000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '15')
        lidar_bp.set_attribute('lower_fov', '-25')
        
        lidar_transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        
        lidar = self.world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self.vehicle
        )
        
        print(f"  ✓ {name}")
        return lidar
    


def main():
    client = carla.Client  ('localhost', 2000)
    client.set_timeout(10.0)
    av = AutonomousVehicle(client)
    world,map = av.spawn_world('Town01')
    vehicle = av.spawn_vehicle('vehicle.mercedes.coupe')
    av.equip_sensors()
    av.vehicle.set_autopilot(True)
    time.sleep(30)

if __name__ == '__main__':
    main()


