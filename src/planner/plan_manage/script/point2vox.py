import os
import numpy as np

class PointCloudVoxelization:
    def __init__(self, input_folder, output_folder, grid_size):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.grid_size = grid_size

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def _read_SemKITTI(self, path, dtype, do_unpack):
        data = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            data = self.unpack(data)
        return data

    def _read_pointcloud_SemKITTI(self, path):
        # Return point cloud semantic kitti with remissions (x, y, z, intensity)
        point_cloud = self._read_SemKITTI(path, dtype=np.float32, do_unpack=False)
        point_cloud = point_cloud.reshape((-1, 4))
        return point_cloud

    def pack(self, array):
        assert array.size % 8 == 0, "The input array size must be divisible by 8."
        array = array.reshape((-1))

        # compressing bit flags.
        compressed = (
            (array[::8] << 7) | (array[1::8] << 6) | (array[2::8] << 5) | (array[3::8] << 4) 
            | (array[4::8] << 3) | (array[5::8] << 2) | (array[6::8] << 1) | array[7::8]
        )

        return np.array(compressed, dtype=np.uint8)

    def compute_voxel_params(self):
        min_coords = np.full(3, np.inf)
        max_coords = np.full(3, -np.inf)

        for file_name in os.listdir(self.input_folder):
            if file_name.endswith('.bin'):
                input_path = os.path.join(self.input_folder, file_name)
                
                # Read point cloud
                point_cloud = self._read_pointcloud_SemKITTI(input_path)
                points_xyz = point_cloud[:, :3]
                
                # Update min_coords and max_coords
                min_coords = np.minimum(min_coords, np.min(points_xyz, axis=0))
                max_coords = np.maximum(max_coords, np.max(points_xyz, axis=0))

        #voxel_origin = min_coords
        # desired_size = max_coords - min_coords
        # voxel_size = desired_size / np.asarray(self.grid_size)
        voxel_size = 0.1
        voxel_origin = np.array([-10, -10, -0.1])
        # print(f"voxel_origin: {voxel_origin}") 
        return voxel_origin, voxel_size


    def voxelization(self):
        # Compute voxel origin and size for the entire sequence
        voxel_origin, voxel_size = self.compute_voxel_params()

        for file_name in os.listdir(self.input_folder):
            if file_name.endswith('.bin'):
                input_path = os.path.join(self.input_folder, file_name)
                output_path = os.path.join(self.output_folder, file_name)
                print(input_path)
                # Read point cloud
                point_cloud = self._read_pointcloud_SemKITTI(input_path)
                points_xyz = point_cloud[:, :3]
                #print(f"points_xyz: {points_xyz}") 
                # Voxelization
                voxel_coords = ((points_xyz - voxel_origin) // voxel_size).astype(int)
                valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < self.grid_size), axis=1)
                voxel_coords = voxel_coords[valid_mask]
                # print("Voxel Coordinates:")
                # for voxel_coord in voxel_coords:
                #     print(voxel_coord)
                # Create an empty voxel grid
                voxel_grid = np.zeros(np.prod(self.grid_size), dtype=np.uint8)

                # Reshape voxel grid before assigning values
                voxel_grid_3d = voxel_grid.reshape(self.grid_size)

                # Set occupied voxels
                voxel_grid_3d[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

                # Convert voxel grid back to a 1D array
                voxel_grid = voxel_grid_3d.flatten()
                
                # Convert voxel grid to bitwise representation
                voxel_grid_bin = self.pack(voxel_grid)

                # Save voxel grid to binary file
                voxel_grid_bin.tofile(output_path)
                
                # Count occupied voxels
                occupied_voxels_count = np.count_nonzero(voxel_grid == 1)

                # Print the file name and occupied voxels count
                print(f"File: {file_name}, Occupied voxels: {occupied_voxels_count}")
                
                
