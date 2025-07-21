import tensorflow as tf
import os


def configTensorflow(useGPU: bool = True):
    """
    Configura o TensorFlow para usar ou não a GPU.

    Parâmetros:
    - useGPU (bool): Define se a GPU será utilizada (True) ou desativada (False).

    Caso a GPU seja desativada, o TensorFlow funcionará exclusivamente na CPU.
    Se a GPU for ativada, o código tenta configurar o uso otimizado do dispositivo.

    Retorna:
    - None
    """

    if not useGPU:
        # Desativa completamente o uso da GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        try:
            # Define para que apenas CPUs estejam visíveis para o TensorFlow
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # Se o TensorFlow já foi inicializado, essa configuração não pode ser alterada
            pass

        # Configura explicitamente o TensorFlow para usar apenas CPUs
        cpus = tf.config.list_physical_devices("CPU")
        if cpus:
            try:
                tf.config.set_visible_devices(cpus, "CPU")
            except RuntimeError:
                pass  # Caso já tenha sido inicializado, não pode modificar os dispositivos

        return

    # Reduz a verbosidade dos logs do TensorFlow para evitar avisos desnecessários
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Define qual GPU será usada (por padrão, usa a GPU de índice 0)
    gpu_num = 0  # Modifique este valor para escolher outra GPU, se disponível

    # Lista todas as GPUs disponíveis
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            # Define a GPU específica a ser usada
            tf.config.set_visible_devices(gpus[gpu_num], "GPU")

            # Ativa o crescimento dinâmico de memória para evitar alocação excessiva
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True)

        except RuntimeError as e:
            print(f"Erro ao configurar a GPU: {e}")

    return
