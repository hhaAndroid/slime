import base64
import io
from multiprocessing.reduction import ForkingPickler
import torch.multiprocessing


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


def deserialize_tensor(data):
    """
    Deserialize a tensor object from serialized data.

    Args:
        data: The serialized tensor data, which can be a base64-encoded string or bytes.

    Returns:
        torch.Tensor: The deserialized tensor.
    """
    b = MultiprocessingSerializer.deserialize(data)
    print(f"after data b : {b}")

    b[0, 2] = 3


if __name__ == '__main__':
    # 验证 cuda 对象是否是 0 copy
    a = torch.randn(2, 3).cuda()

    print(f"Original data a : {a}")
    serialized = MultiprocessingSerializer.serialize(a, output_str=True)
    print(f"Serialized data type: {type(serialized)}")

    # cuda ipc 只能在跨进程中使用，没有下面的多进程会报错
    torch.multiprocessing.set_start_method("spawn")
    new_process = torch.multiprocessing.Process(target=deserialize_tensor, args=(serialized,))
    new_process.start()
    new_process.join()

    # 从打印可以看出， cuda ipc 传递的是 tensor meta 信息而非数据本身
    # 反序列化后的也是 meta data，然后可以直接重建，0 copy
    print(f"after data a : {a}")

