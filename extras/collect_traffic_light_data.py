#!/usr/bin/env python3
"""
collect_traffic_light_data.py

Collects traffic light images from CARLA at various distances.
Uses AutonomousVehicle class for vehicle and sensor setup.

Output structure:
  output_dir/
    narrow/
      40m/
      30m/
      20m/
      10m/
    wide/
      40m/
      30m/
      20m/
      10m/

Usage:
  python collect_traffic_light_data.py --output ./traffic_data --duration 300
"""

from vehicle.autonomous_vehicle import AutonomousVehicle
import carla
import cv2
import numpy as np
import os
import time
from datetime import datetime
from collections import defaultdict
import argparse
import pygame


class TrafficLightDataCollector:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
        # Create folder structure
        self.cameras = ['narrow', 'wide']
        self.distances = ['40m', '30m', '20m', '10m']
        
        for cam in self.cameras:
            for dist in self.distances:
                os.makedirs(os.path.join(output_dir, cam, dist), exist_ok=True)
        
        # Image counters per folder
        self.counters = defaultdict(int)
        
        # Track captured traffic lights to avoid duplicates
        self.captured = defaultdict(bool)
        
        # Current frames
        self.frame_narrow = None
        self.frame_wide = None
        
        # Stats
        self.stats = defaultdict(int)
    
    def get_distance_bucket(self, distance):
        """Get distance bucket for a given distance"""
        if distance <= 12:
            return '10m'
        elif distance <= 25:
            return '20m'
        elif distance <= 35:
            return '30m'
        elif distance <= 45:
            return '40m'
        return None
    
    def narrow_callback(self, image):
        """Callback for narrow camera"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.frame_narrow = array[:, :, :3].copy()
    
    def wide_callback(self, image):
        """Callback for wide camera"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.frame_wide = array[:, :, :3].copy()
    
    def save_image(self, camera, distance_bucket):
        """Save image to appropriate folder"""
        if camera == 'narrow' and self.frame_narrow is not None:
            frame = self.frame_narrow
        elif camera == 'wide' and self.frame_wide is not None:
            frame = self.frame_wide
        else:
            return None
        
        folder = os.path.join(self.output_dir, camera, distance_bucket)
        idx = self.counters[(camera, distance_bucket)]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"traffic_light_{timestamp}_{idx:06d}.png"
        filepath = os.path.join(folder, filename)
        
        cv2.imwrite(filepath, frame)
        self.counters[(camera, distance_bucket)] += 1
        self.stats[(camera, distance_bucket)] += 1
        
        return filename
    
    def run(self, duration=300, town='Town01'):
        """Run data collection"""
        print(f"Starting traffic light data collection...")
        print(f"Output: {self.output_dir}")
        print(f"Duration: {duration} seconds")
        print(f"Town: {town}")
        
        # Connect to CARLA and initialize vehicle using AV class
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        av = AutonomousVehicle(client)
        av.spawn_world(town)
        av.spawn_vehicle('vehicle.tesla.model3')
        av.equip_sensors()
        
        # Get references
        world = av.world
        spectator = world.get_spectator()
        
        # Set world to sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Listen to cameras
        av.sensors['front_narrow'].listen(self.narrow_callback)
        av.sensors['front_wide'].listen(self.wide_callback)
        
        # Initialize pygame for keyboard input
        pygame.init()
        screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("CARLA Manual Control - Keep this window focused!")
        
        # Manual control instructions
        print("\n" + "="*50)
        print("MANUAL CONTROL MODE - WASD Keys")
        print("="*50)
        print("W - Accelerate")
        print("S - Brake / Reverse")
        print("A - Steer Left")
        print("D - Steer Right")
        print("SPACE - Handbrake")
        print("Q - Quit")
        print("="*50)
        print("\nKEEP THE PYGAME WINDOW FOCUSED FOR CONTROLS!")
        print("Collecting data... Press Q or Ctrl+C to stop.\n")
        
        start_time = time.time()
        
        try:
            running = True
            while running and time.time() - start_time < duration:
                # Handle pygame events and keyboard input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                keys = pygame.key.get_pressed()
                
                # Quit on Q
                if keys[pygame.K_q]:
                    running = False
                    break
                
                # Create vehicle control
                control = carla.VehicleControl()
                
                # Throttle/Brake (W/S)
                if keys[pygame.K_w]:
                    control.throttle = 0.7
                    control.reverse = False
                elif keys[pygame.K_s]:
                    control.brake = 0.5
                    # Hold S longer for reverse
                    velocity = av.vehicle.get_velocity()
                    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                    if speed < 0.1:
                        control.throttle = 0.5
                        control.reverse = True
                        control.brake = 0.0
                
                # Steering (A/D)
                if keys[pygame.K_a]:
                    control.steer = -0.5
                elif keys[pygame.K_d]:
                    control.steer = 0.5
                
                # Handbrake (SPACE)
                if keys[pygame.K_SPACE]:
                    control.hand_brake = True
                
                # Apply control
                av.vehicle.apply_control(control)
                
                # Update pygame display
                screen.fill((30, 30, 30))
                font = pygame.font.Font(None, 36)
                text = font.render("WASD to drive | Q to quit", True, (255, 255, 255))
                screen.blit(text, (50, 80))
                pygame.display.flip()
                
                world.tick()
                time.sleep(0.02)
                
                # Update spectator to front camera view (like wide camera)
                vehicle_transform = av.vehicle.get_transform()
                forward = vehicle_transform.get_forward_vector()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(
                        x=forward.x * 2.5,
                        y=forward.y * 2.5,
                        z=1.5
                    ),
                    carla.Rotation(pitch=-5, yaw=vehicle_transform.rotation.yaw)
                )
                spectator.set_transform(spectator_transform)
                
                if self.frame_narrow is None or self.frame_wide is None:
                    continue
                
                # Get vehicle location
                vehicle_loc = av.vehicle.get_location()
                
                # Find nearby traffic lights
                traffic_lights = world.get_actors().filter('traffic.traffic_light')
                
                for tl in traffic_lights:
                    tl_loc = tl.get_location()
                    distance = vehicle_loc.distance(tl_loc)
                    
                    # Get distance bucket
                    bucket = self.get_distance_bucket(distance)
                    if bucket is None:
                        continue
                    
                    tl_id = tl.id
                    
                    # Check if traffic light is in front of vehicle
                    vehicle_transform = av.vehicle.get_transform()
                    forward = vehicle_transform.get_forward_vector()
                    to_tl = carla.Location(
                        x=tl_loc.x - vehicle_loc.x,
                        y=tl_loc.y - vehicle_loc.y,
                        z=0
                    )
                    
                    # Dot product to check if in front
                    dot = forward.x * to_tl.x + forward.y * to_tl.y
                    if dot < 0:
                        continue  # Behind vehicle
                    
                    # Save for each camera if not already captured
                    for camera in self.cameras:
                        key = (tl_id, bucket, camera)
                        if not self.captured[key]:
                            filename = self.save_image(camera, bucket)
                            if filename:
                                self.captured[key] = True
                                state = tl.get_state()
                                print(f"[{camera}/{bucket}] Saved: {filename} (state: {state})")
                
                # Print progress every 30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    total = sum(self.stats.values())
                    print(f"\n--- Progress: {int(elapsed)}s / {duration}s | Total images: {total} ---\n")
        
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
        
        finally:
            # Cleanup - stop and destroy sensors
            print("\nCleaning up...")
            pygame.quit()
            for name, sensor in av.sensors.items():
                sensor.stop()
                sensor.destroy()
            av.vehicle.destroy()
            
            # Reset sync mode
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        # Print summary
        print("\n" + "="*50)
        print("COLLECTION SUMMARY")
        print("="*50)
        
        total = 0
        for cam in self.cameras:
            print(f"\n{cam.upper()} camera:")
            for dist in self.distances:
                count = self.stats[(cam, dist)]
                total += count
                print(f"  {dist}: {count} images")
        
        print(f"\nTotal: {total} images")
        print(f"Output: {self.output_dir}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Collect traffic light images from CARLA')
    parser.add_argument('--output', type=str, default='./traffic_data6',
                        help='Output directory')
    parser.add_argument('--duration', type=int, default=3000,
                        help='Collection duration in seconds')
    parser.add_argument('--town', type=str, default='Town04',
                        help='CARLA town to use')
    args = parser.parse_args()
    
    collector = TrafficLightDataCollector(args.output)
    collector.run(duration=args.duration, town=args.town)


if __name__ == '__main__':
    main()
