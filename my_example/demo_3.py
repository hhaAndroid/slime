import base64
import io
from multiprocessing.reduction import ForkingPickler
import torch.multiprocessing
import ray


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = base64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = base64.b64decode(data)

        return ForkingPickler.loads(data)


@ray.remote(
    num_gpus=1,
    runtime_env={
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",  # 这个非常关键, 防止每个 actor 只能看到一个 device，导致 torch.cuda.set_device 报错
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        }
    },
)
class RayActorA:
    def __init__(self):
        torch.cuda.set_device(f"cuda:{ray.get_gpu_ids()[0]}")
        self.data = torch.randn(2, 3).cuda()
        print("[RayActorA] gpu_ids:", ray.get_gpu_ids()[0], self.data)

    def serialize_tensor(self):
        serialized_data = MultiprocessingSerializer.serialize(self.data, output_str=True)
        return serialized_data

    def get_data(self):
        return self.data


@ray.remote(
    num_gpus=1,
    runtime_env={
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        }
    },
)
class RayActorB:
    def __init__(self):
        torch.cuda.set_device(f"cuda:{ray.get_gpu_ids()[0]}")
        self.data = torch.randn(2, 3).cuda()
        print("[RayActorB] gpu_ids:", ray.get_gpu_ids()[0], self.data)

    def deserialize_tensor(self, serialized_data):
        print("[RayActorB] before:", self.data)
        deserialized_tensor = MultiprocessingSerializer.deserialize(serialized_data)
        self.data = deserialized_tensor

        self.data[0, 2] = 100
        print("[RayActorB] after:", self.data)

    def get_data(self):
        return self.data


# 通过该案例可以证明： cuda ipc 可以在同一个节点的不同 gpu 直接直接共享对象，
# 例如将 cuda0 对象发送给 cuda1 ，并且在那边反序列化后得到的数据就变成 cuda0 了
if __name__ == '__main__':
    ray.init()

    actor_a = RayActorA.remote()
    actor_b = RayActorB.remote()

    serialized_data = ray.get(actor_a.serialize_tensor.remote())
    ray.get(actor_b.deserialize_tensor.remote(serialized_data))

    # 打印
    data_a = ray.get(actor_a.get_data.remote())
    data_b = ray.get(actor_b.get_data.remote())
    print(f"Data in Actor A: {data_a}, Data in Actor B: {data_b}")
