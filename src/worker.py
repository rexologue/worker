import asyncio
from typing import Any

from aiosock.typs import Message
from aiosock import ApiBlueprint, ClientSecureSocket

from models import ModelHost, ModelManager, ModelSpec
from utils.enums import EngineType
from utils.get_port import get_safe_free_port
from utils.local_characteristics import get_inference_env_info


############################
# WORKER CLASS REALISATION #
############################


class Worker:
    def __init__(self, socket: ClientSecureSocket, token: str = "") -> None:
        """NAL NET Worker.

        Подключается к брокеру, принимает команды и выполняет задачи на локальной машине.
        """
        self.worker = socket
        self.worker.api = self._build_api()

        self.token = token
        self.model_host: ModelHost | None = None
        self.model_manager = ModelManager("")

    @classmethod
    async def create(
        cls, host: str, port: int | None = None, token: str = ""
    ) -> "Worker":
        socket = await ClientSecureSocket.create(host, port, token)
        return cls(socket, token)

    #######
    # API #
    #######

    def _build_api(self) -> ApiBlueprint:
        api = ApiBlueprint()

        # Отключение по инициативе брокера
        @api.handler("BYE")
        async def bye(msg: Message[dict, ClientSecureSocket]) -> None:
            reason = msg.payload.get("reason", "")
            print(f"[WORKER] Disconnected by broker: {reason}")
            asyncio.create_task(self._disconnect())

        # Загрузка модели и подготовка рабочего окружения
        @api.handler("RUN_WORKER")
        async def run_worker(msg: Message[dict, ClientSecureSocket]) -> None:
            engine_type = msg.payload.get("engine_type")
            model_id = msg.payload.get("model_id")
            repo_id = msg.payload.get("repo_id")
            quant = msg.payload.get("quantization")
            src = msg.payload.get("source")
 
            try:
                engine = next((e for e in EngineType if e.name == engine_type), None)
            except Exception:
                engine = None

            if engine is None or not repo_id:
                print("[WORKER] Bad config in RUN_WORKER, disconnecting…")
                asyncio.create_task(
                    self._disconnect(
                        message="Disconnect due to bad config in RUN_WORKER"
                    )
                )
                return

            # Скачиваем модель и стартуем хост
            self.model_manager.add(
                name=model_id, spec=ModelSpec(
                    source=src,
                    repo_id=repo_id,
                    gguf_quant=quant
                )
            )

            path = self.model_manager.get_model(model_id, quant)
            random_port = get_safe_free_port()

            self.model_host = ModelHost(
                engine_type=engine,
                model_path=path,
                port=random_port,
            )

            try:
                # запуск модели может быть блокирующим — уводим в пул
                await asyncio.to_thread(self.model_host.start)
            except Exception as e:
                print(f"[WORKER] Model host failed: {e}")
                asyncio.create_task(
                    self._disconnect(
                        message=f"Cannot load model in RUN_WORKER: {e}"
                    )
                )
                return

            # Сообщаем брокеру, что готовы
            asyncio.create_task(
                self.worker.send("WORKER_READY", {"status": "OK"})
            )
            print(f"[WORKER] Ready! Model is running on port {random_port}")

        # Выполнение задачи генерации
        @api.handler("GENERATE")
        async def generate(msg: Message[dict, ClientSecureSocket]) -> dict:
            """
            ВАЖНО: возвращаем словарь как результат, чтобы Библиотека
            AioSock автоматически сформировала пакет "RESPONSE" c тем же id.
            Это позволяет брокеру ждать ответ через Future по message_id.
            """

            task = msg.payload.get("task")
            worker_id = msg.payload.get("worker_id")
            if not isinstance(task, dict) or not worker_id:
                return {"error": "bad task payload"}

            client_id = task.get("client_id")
            kind = task.get("kind")
            print(f"[WORKER] Got task: kind={kind}")

            if kind == "text":
                prompt = task.get("prompt")
                if prompt is None:
                    result = ["None", 0]
                else:
                    if self.model_host is None:
                        return {"error": "model is not loaded"}
                    # синхронный инференс — уводим в пул, если нужно
                    result = await asyncio.to_thread(self.model_host.generate, prompt)
            else:
                result = [f"Unknown kind of task: {kind}", 0]

            return {
                "result": result,
                "client_id": client_id,
                "worker_id": worker_id,
            }

        return api

    ###################
    # UTILITY METHODS #
    ###################

    async def _send_hello(self) -> None:
        """
        HELLO_FROM_WORKER — отправляем токен и характеристики.
        """
        info = get_inference_env_info()
        await self.worker.send(
            command="HELLO_FROM_WORKER",
            data={
                "token": self.token,
                "info": info,
            },
        )

    async def _disconnect(self, delay: float = 2.0, message: str | None = None) -> None:
        if message is not None:
            await self.worker.send(
                command="DISCONNECT_WORKER",
                data={"reason": message},
            )

        await asyncio.sleep(delay)
        await self.stop()

    ############################
    # WORKER START/STOP METHOD #
    ############################

    async def start(self) -> None:
        """
        ВАЖНО: ClientSecureSocket.start() — БЛОКИРУЮЩИЙ цикл.
        Поэтому запускаем его фоном, дожидаемся подключения, шлём HELLO,
        а затем ждём завершения фоновой задачи.
        """
        # старт клиента (фоновая задача)
        start_task = asyncio.create_task(self.worker.start())

        # дожидаемся установки соединения (см. ClientSecureSocket.wait_connection)
        await self.worker.wait_connection()

        # шлём HELLO строго после установления защищённого канала
        await self._send_hello()

        # далее блокируемся, пока клиент работает
        await start_task

    async def stop(self) -> None:
        if self.model_host is not None:
            try:
                self.model_host.stop()
            except Exception:
                pass
        await self.worker.stop()

    def is_alive(self) -> bool:
        return self.worker.is_alive()


##############
# RUN WORKER #
##############


async def run_worker(server_host: str, server_port: int, token: str) -> None:
    worker = await Worker.create(server_host, server_port, token)

    try:
        await worker.start()
    finally:
        if worker.is_alive():
            await worker.stop()
