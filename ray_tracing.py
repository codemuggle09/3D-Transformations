"""
Vulkan Ray Tracing Implementation with Acceleration Structures
Requires: vulkan, glfw, numpy
Install: pip install vulkan glfw numpy
"""

import vulkan as vk
import glfw
import numpy as np
import struct
from ctypes import *

class VulkanRayTracer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.instance = None
        self.device = None
        self.physical_device = None
        self.queue = None
        self.command_pool = None
        self.pipeline = None
        self.pipeline_layout = None
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_set = None
        
        # Ray tracing specific
        self.blas = None  # Bottom-level acceleration structure
        self.tlas = None  # Top-level acceleration structure
        self.sbt_buffer = None  # Shader binding table
        
    def initialize(self):
        """Initialize Vulkan and ray tracing extensions"""
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        
        # Check for ray tracing support
        self._create_instance()
        self._select_physical_device()
        self._create_device()
        self._create_command_pool()
        
    def _create_instance(self):
        """Create Vulkan instance with ray tracing extensions"""
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Ray Tracer",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_2
        )
        
        extensions = [
            vk.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
        ]
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        
        self.instance = vk.vkCreateInstance(create_info, None)
        
    def _select_physical_device(self):
        """Select GPU with ray tracing support"""
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        
        for device in devices:
            props = vk.vkGetPhysicalDeviceProperties(device)
            features = vk.vkGetPhysicalDeviceFeatures(device)
            
            # Check for ray tracing support
            ext_props = vk.vkEnumerateDeviceExtensionProperties(device, None)
            has_rt = any(ext.extensionName == vk.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME 
                        for ext in ext_props)
            
            if has_rt:
                self.physical_device = device
                print(f"Selected device: {props.deviceName}")
                return
                
        raise Exception("No device with ray tracing support found")
        
    def _create_device(self):
        """Create logical device with ray tracing extensions"""
        queue_family_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self.physical_device
        )
        
        queue_family_index = 0
        for i, props in enumerate(queue_family_props):
            if props.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                queue_family_index = i
                break
        
        queue_priority = [1.0]
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=queue_priority
        )
        
        # Ray tracing extensions
        extensions = [
            vk.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk.VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            vk.VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            vk.VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            vk.VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
        ]
        
        # Enable ray tracing features
        rt_pipeline_features = vk.VkPhysicalDeviceRayTracingPipelineFeaturesKHR(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            rayTracingPipeline=vk.VK_TRUE
        )
        
        as_features = vk.VkPhysicalDeviceAccelerationStructureFeaturesKHR(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            pNext=rt_pipeline_features,
            accelerationStructure=vk.VK_TRUE
        )
        
        device_features = vk.VkPhysicalDeviceFeatures2(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            pNext=as_features
        )
        
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=device_features,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        
        self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, queue_family_index, 0)
        
    def _create_command_pool(self):
        """Create command pool for recording commands"""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=0,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        
    def create_acceleration_structures(self, vertices, indices, instances):
        """
        Create bottom-level and top-level acceleration structures
        
        vertices: List of vertex positions [(x,y,z), ...]
        indices: List of triangle indices [i1,i2,i3, ...]
        instances: List of instance transforms and geometry references
        """
        # Create vertex and index buffers
        vertex_buffer = self._create_buffer(
            vertices, 
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        )
        
        index_buffer = self._create_buffer(
            indices,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        )
        
        # Build BLAS (Bottom-level acceleration structure)
        self.blas = self._build_blas(vertex_buffer, index_buffer, len(indices) // 3)
        
        # Build TLAS (Top-level acceleration structure)
        self.tlas = self._build_tlas(instances)
        
    def _build_blas(self, vertex_buffer, index_buffer, triangle_count):
        """Build bottom-level acceleration structure for geometry"""
        # Define geometry
        geometry = vk.VkAccelerationStructureGeometryKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR
        )
        
        # Triangle geometry data
        triangles = vk.VkAccelerationStructureGeometryTrianglesDataKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            vertexFormat=vk.VK_FORMAT_R32G32B32_SFLOAT,
            vertexData=self._get_buffer_device_address(vertex_buffer),
            vertexStride=12,  # 3 floats
            maxVertex=triangle_count * 3,
            indexType=vk.VK_INDEX_TYPE_UINT32,
            indexData=self._get_buffer_device_address(index_buffer)
        )
        
        geometry.geometry.triangles = triangles
        
        # Build geometry info
        build_info = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            geometryCount=1,
            pGeometries=[geometry]
        )
        
        # Get size requirements
        size_info = vk.vkGetAccelerationStructureBuildSizesKHR(
            self.device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            build_info,
            [triangle_count]
        )
        
        # Create acceleration structure buffer
        as_buffer = self._create_buffer(
            None,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            size_info.accelerationStructureSize
        )
        
        # Create acceleration structure
        as_create_info = vk.VkAccelerationStructureCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            buffer=as_buffer,
            size=size_info.accelerationStructureSize,
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        )
        
        blas = vk.vkCreateAccelerationStructureKHR(self.device, as_create_info, None)
        
        # Build acceleration structure
        self._build_acceleration_structure(build_info, blas, size_info)
        
        return blas
        
    def _build_tlas(self, instances):
        """Build top-level acceleration structure for scene instances"""
        # Create instance buffer
        instance_data = []
        for inst in instances:
            transform = inst['transform']  # 3x4 matrix
            blas_address = self._get_as_device_address(inst['blas'])
            
            instance = vk.VkAccelerationStructureInstanceKHR(
                transform=transform,
                instanceCustomIndex=inst.get('custom_index', 0),
                mask=0xFF,
                instanceShaderBindingTableRecordOffset=0,
                flags=vk.VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                accelerationStructureReference=blas_address
            )
            instance_data.append(instance)
        
        instance_buffer = self._create_buffer(
            instance_data,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        )
        
        # Build TLAS
        geometry = vk.VkAccelerationStructureGeometryKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=vk.VK_GEOMETRY_TYPE_INSTANCES_KHR,
            flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR
        )
        
        instances_data = vk.VkAccelerationStructureGeometryInstancesDataKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            arrayOfPointers=vk.VK_FALSE,
            data=self._get_buffer_device_address(instance_buffer)
        )
        
        geometry.geometry.instances = instances_data
        
        build_info = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            geometryCount=1,
            pGeometries=[geometry]
        )
        
        # Get size and create TLAS
        size_info = vk.vkGetAccelerationStructureBuildSizesKHR(
            self.device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            build_info,
            [len(instances)]
        )
        
        as_buffer = self._create_buffer(
            None,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            size_info.accelerationStructureSize
        )
        
        as_create_info = vk.VkAccelerationStructureCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            buffer=as_buffer,
            size=size_info.accelerationStructureSize,
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
        )
        
        tlas = vk.vkCreateAccelerationStructureKHR(self.device, as_create_info, None)
        
        self._build_acceleration_structure(build_info, tlas, size_info)
        
        return tlas
        
    def create_ray_tracing_pipeline(self):
        """Create ray tracing pipeline with shaders"""
        # Load shaders (SPIR-V bytecode)
        rgen_shader = self._load_shader("raygen.rgen.spv")
        rmiss_shader = self._load_shader("miss.rmiss.spv")
        rchit_shader = self._load_shader("closesthit.rchit.spv")
        
        # Shader stages
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                module=rgen_shader,
                pName="main"
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_MISS_BIT_KHR,
                module=rmiss_shader,
                pName="main"
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                module=rchit_shader,
                pName="main"
            )
        ]
        
        # Shader groups
        groups = [
            # Ray generation group
            vk.VkRayTracingShaderGroupCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type=vk.VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                generalShader=0,
                closestHitShader=vk.VK_SHADER_UNUSED_KHR,
                anyHitShader=vk.VK_SHADER_UNUSED_KHR,
                intersectionShader=vk.VK_SHADER_UNUSED_KHR
            ),
            # Miss group
            vk.VkRayTracingShaderGroupCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type=vk.VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                generalShader=1,
                closestHitShader=vk.VK_SHADER_UNUSED_KHR,
                anyHitShader=vk.VK_SHADER_UNUSED_KHR,
                intersectionShader=vk.VK_SHADER_UNUSED_KHR
            ),
            # Hit group
            vk.VkRayTracingShaderGroupCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type=vk.VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                generalShader=vk.VK_SHADER_UNUSED_KHR,
                closestHitShader=2,
                anyHitShader=vk.VK_SHADER_UNUSED_KHR,
                intersectionShader=vk.VK_SHADER_UNUSED_KHR
            )
        ]
        
        # Pipeline creation
        pipeline_info = vk.VkRayTracingPipelineCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            stageCount=len(stages),
            pStages=stages,
            groupCount=len(groups),
            pGroups=groups,
            maxPipelineRayRecursionDepth=2,
            layout=self.pipeline_layout
        )
        
        self.pipeline = vk.vkCreateRayTracingPipelinesKHR(
            self.device,
            vk.VK_NULL_HANDLE,
            vk.VK_NULL_HANDLE,
            1,
            [pipeline_info],
            None
        )[0]
        
        # Create shader binding table
        self._create_shader_binding_table(groups)
        
    def _create_shader_binding_table(self, groups):
        """Create shader binding table for ray tracing"""
        rt_props = vk.vkGetPhysicalDeviceRayTracingPipelinePropertiesKHR(
            self.physical_device
        )
        
        handle_size = rt_props.shaderGroupHandleSize
        handle_alignment = rt_props.shaderGroupHandleAlignment
        
        # Get shader group handles
        handles_size = len(groups) * handle_size
        handles = vk.vkGetRayTracingShaderGroupHandlesKHR(
            self.device,
            self.pipeline,
            0,
            len(groups),
            handles_size
        )
        
        # Create SBT buffer
        sbt_size = len(groups) * handle_alignment
        self.sbt_buffer = self._create_buffer(
            handles,
            vk.VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
            vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            sbt_size
        )
        
    def trace_rays(self, output_image):
        """Execute ray tracing"""
        cmd_buffer = self._begin_single_time_commands()
        
        # Bind pipeline
        vk.vkCmdBindPipeline(
            cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            self.pipeline
        )
        
        # Bind descriptor sets
        vk.vkCmdBindDescriptorSets(
            cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            self.pipeline_layout,
            0, 1, [self.descriptor_set],
            0, None
        )
        
        # Shader binding table regions
        rt_props = vk.vkGetPhysicalDeviceRayTracingPipelinePropertiesKHR(
            self.physical_device
        )
        
        handle_size_aligned = self._align_up(
            rt_props.shaderGroupHandleSize,
            rt_props.shaderGroupHandleAlignment
        )
        
        sbt_address = self._get_buffer_device_address(self.sbt_buffer)
        
        raygen_region = vk.VkStridedDeviceAddressRegionKHR(
            deviceAddress=sbt_address,
            stride=handle_size_aligned,
            size=handle_size_aligned
        )
        
        miss_region = vk.VkStridedDeviceAddressRegionKHR(
            deviceAddress=sbt_address + handle_size_aligned,
            stride=handle_size_aligned,
            size=handle_size_aligned
        )
        
        hit_region = vk.VkStridedDeviceAddressRegionKHR(
            deviceAddress=sbt_address + 2 * handle_size_aligned,
            stride=handle_size_aligned,
            size=handle_size_aligned
        )
        
        callable_region = vk.VkStridedDeviceAddressRegionKHR()
        
        # Trace rays
        vk.vkCmdTraceRaysKHR(
            cmd_buffer,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
            self.width,
            self.height,
            1
        )
        
        self._end_single_time_commands(cmd_buffer)
        
    def _create_buffer(self, data, usage, size=None):
        """Helper to create Vulkan buffer"""
        if size is None and data is not None:
            size = len(data) * data.itemsize if hasattr(data, 'itemsize') else len(data)
        
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        
        # Allocate and bind memory
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=self._find_memory_type(
                mem_reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
        )
        
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        # Copy data if provided
        if data is not None:
            # Create staging buffer and copy
            pass  # Implementation details omitted for brevity
        
        return buffer
        
    def _load_shader(self, filename):
        """Load SPIR-V shader module"""
        with open(filename, 'rb') as f:
            code = f.read()
        
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code
        )
        
        return vk.vkCreateShaderModule(self.device, create_info, None)
        
    def cleanup(self):
        """Cleanup Vulkan resources"""
        if self.device:
            vk.vkDeviceWaitIdle(self.device)
            
            if self.pipeline:
                vk.vkDestroyPipeline(self.device, self.pipeline, None)
            if self.pipeline_layout:
                vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
            if self.blas:
                vk.vkDestroyAccelerationStructureKHR(self.device, self.blas, None)
            if self.tlas:
                vk.vkDestroyAccelerationStructureKHR(self.device, self.tlas, None)
            
            vk.vkDestroyDevice(self.device, None)
        
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
        
        glfw.terminate()


# Example usage
if __name__ == "__main__":
    raytracer = VulkanRayTracer(800, 600)
    
    try:
        raytracer.initialize()
        
        # Define scene geometry
        vertices = np.array([
            # Triangle 1
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 0.0,  1.0, 0.0],
            # Add more geometry...
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2], dtype=np.uint32)
        
        # Create acceleration structures
        raytracer.create_acceleration_structures(vertices, indices, [
            {
                'blas': raytracer.blas,
                'transform': np.eye(3, 4, dtype=np.float32),
                'custom_index': 0
            }
        ])
        
        # Create ray tracing pipeline
        raytracer.create_ray_tracing_pipeline()
        
        # Render
        output_image = None  # Create output image buffer
        raytracer.trace_rays(output_image)
        
        print("Ray tracing completed successfully!")
        
    finally:
        raytracer.cleanup()