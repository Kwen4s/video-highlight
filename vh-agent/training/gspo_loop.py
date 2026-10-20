"""Give each video processor call its own mutable, plain Python options."""

from omegaconf import OmegaConf
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class VideoPromptClient:
    def __init__(self, client, prompt_ids):
        self.client = client
        self.prompt_ids = prompt_ids

    async def generate(self, request_id, **kwargs):
        # vLLM expands raw video placeholders itself. The loop keeps HF-expanded
        # ids for learning; sending those to vLLM would expand the video twice.
        kwargs["prompt_ids"] = self.prompt_ids
        return await self.client.generate(request_id=request_id, **kwargs)


class VideoIntervalLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params, **kwargs):
        text = self.processor.apply_chat_template(
            list(kwargs["raw_prompt"]),
            tokenize=False,
            add_generation_prompt=True,
            **self.apply_chat_template_kwargs,
        )
        raw_ids = self.tokenizer.encode(text, add_special_tokens=False)
        client = self.server_manager
        self.server_manager = VideoPromptClient(client, raw_ids)
        try:
            return await super().run(sampling_params, **kwargs)
        finally:
            self.server_manager = client

    def _get_mm_processor_kwargs(self, audio_data=None):
        return OmegaConf.to_container(self.mm_processor_kwargs, resolve=True)

    async def ct_build_initial_tokens(self, *args, **kwargs):
        # Transformers consumes nested videos_kwargs with pop(). Never share it
        # with Hydra config, another trajectory or the training preprocessing.
        self.continuous_token_builder.mm_processor_kwargs = self._get_mm_processor_kwargs()
        return await super().ct_build_initial_tokens(*args, **kwargs)
