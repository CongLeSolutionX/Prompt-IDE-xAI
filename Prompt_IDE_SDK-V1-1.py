"""PromptIDE SDK version 1.1."""

import asyncio
import contextlib
import contextvars
import dataclasses
import random
import time
import uuid
from typing import Any, Optional, Sequence, Union

import js
import pyodide

_USER = 1
_MODEL = 2


@dataclasses.dataclass(frozen=True)
class Token:
    """A token is an element of our vocabulary that has a unique index and string representation.

    A token can either be sampled from a model or provided by the user (i.e. prompted). If the token
    comes from the mode, we may have additional metadata such as its sampling probability, the
    attention pattern used when sampling the token, and alternative tokens.
    """

    # The integer representation of the token. Corresponds to its index in the vocabulary.
    token_id: int
    # The string representation of the token. Corresponds to its value in the vocabulary.
    token_str: str
    # If this token was sampled, the token sampling probability. 0 if not sampled.
    prob: float
    # If this token was sampled, alternative tokens that could have been sampled instead.
    top_k: list["Token"]
    # If this token was sampled with the correct options, the token's attention pattern. The array
    # contains one value for every token in the context.
    attn_weights: list[float]
    # 1 if this token was created by a user and 2 if it was created by model.
    token_type: int

    @classmethod
    def from_proto_dict(cls, values: dict) -> "Token":
        """Converts the protobuffer dictionary to a `Token` instance."""
        return Token(
            token_id=values["finalLogit"]["tokenId"],
            token_str=values["finalLogit"]["stringToken"],
            prob=values["finalLogit"]["prob"],
            top_k=[
                Token.from_proto_dict(
                    {"finalLogit": l, "topK": [], "attention": [], "tokenType": _MODEL}
                )
                for l in values["topK"]
            ],
            attn_weights=values["attention"],
            token_type=values["tokenType"],
        )


async def user_input(text: str) -> str | None:
    """Asks the user to enter something into the text field shown in the completion dialog.

    Args:
        text: The placeholder text displayed in the text field before the user enters a response.

    Returns:
        A string if the user actually entered some text and `None` if the user pressed `cancel`.
    """
    args = pyodide.ffi.create_proxy(str(text))
    response = await js.userInput(args)
    response = response.to_py()

    if "cancelled" in response:
        return None
    return response["text"]


@dataclasses.dataclass
class SampleResult:
    """Holds the results of a sampling call."""

    # The actual request made to the sampling API. Note that these fields may be unstable and are
    # subject to change in the future.
    request: dict = dataclasses

    # The number of tokens sampled.
    tokens: list[Token] = dataclasses.field(default_factory=list)
    # When sampling was started.
    start_time: float = dataclasses.field(default_factory=time.time)
    # Time when the first token was added.
    first_token_time: Optional[float] = None
    # When sampling finished.
    end_time: Optional[float] = None

    def as_string(self) -> str:
        """Returns a string representation of this context."""
        return "".join(t.token_str for t in self.tokens)

    def append(self, token: Token):
        """Adds a token to the result and reports progress in the terminal."""
        self.tokens.append(token)
        self.end_time = time.time()
        if len(self.tokens) == 1:
            self.first_token_time = time.time()
            duration = (self.first_token_time - self.start_time) * 1000
            print(f"Sampled first token after {duration:1.2f}ms.")
        elif (len(self.tokens) + 1) % 10 == 0:
            self.print_progress()

    def print_progress(self):
        """Prints the sampling progress to stdout."""
        if len(self.tokens) > 1:
            duration = self.end_time - self.first_token_time
            speed = (len(self.tokens) - 1) / duration
            print(f"Sampled {len(self.tokens)} tokens. {speed:1.2f} tokens/s")


def _parse_input_token(token: Union[int, str]) -> dict:
    """Converts the argument to an `InputToken` proto."""
    if isinstance(token, int):
        return {"tokenId": token}
    else:
        return {"stringToken": token}


@dataclasses.dataclass
class Context:
    """A context is a sequence of tokens that are used as prompt when sampling from the model."""

    # The context ID.
    context_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    # The body of this context is a sequence of tokens and child-contexts. The reasons we use a
    # joint body field instead of separate fields is that we want to render the child contexts
    # relative to the tokens of the parent context.
    body: list[Union[Token, "Context"]] = dataclasses.field(default_factory=list)
    # The parent context if this is not the root context.
    parent: Optional["Context"] = None
    # The seed used for the next call to `sample`.
    next_rng_seed: int = 0
    # Name of the model to use. The model name is tied to the context because different models can
    # use different tokenizers.
    model_name: str = ""

    # If this context has been manually entered, the reset token to reset the global context
    # variable.
    _reset_token: Any = None

    def __post_init__(self):
        """Sends this context to the UI thread to be displayed in the rendering dialogue."""
        if self.parent is not None:
            self.parent.body.append(self)

        request = {
            "contextId": self.context_id,
            "parent": self.parent.context_id if self.parent else "",
        }
        asyncio.get_event_loop().run_until_complete(
            js.createContext(pyodide.ffi.create_proxy(request))
        )

    def select_model(self, model_name: str):
        """Selects the model name for this context.

        The model name can only be set before any tokens have been added to this context.

        Args:
            model_name: Name of the model to use.
        """
        if self.tokens:
            raise RuntimeError(
                "Cannot change the model name of a non-empty context. A context "
                "stores token sequences and different models may use different "
                "tokenizers. Hence, using tokens across models leads to undefined "
                "behavior. If you want to use multiple models in the same prompt, "
                "consider using a @prompt_fn."
            )
        self.model_name = model_name

    async def _tokenize(self, text: str) -> list[dict]:
        """Same as `tokenize` but returns the raw proto dicts."""
        # Nothing to do if the text is empty.
        if not text:
            return []
        print(f"Tokenizing prompt with {len(text)} characters.")
        result = await js.tokenize(
            pyodide.ffi.create_proxy(
                {
                    "text": text,
                    "modelName": self.model_name,
                }
            )
        )
        result = result.to_py()
        compression = (1 - len(result) / len(text)) * 100
        print(
            f"Tokenization done. {len(result)} tokens detected (Compression of {compression:.1f}%)."
        )

        return result

    async def tokenize(self, text: str) -> list[Token]:
        """Tokenizes the given text and returns a list of individual tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens. The log probability on the logit is initialized to 0.
        """
        result = await self._tokenize(text)
        return [Token.from_proto_dict(d) for d in result]

    @property
    def tokens(self) -> Sequence[Token]:
        """Returns the tokens stored in this context."""
        return [t for t in self.body if isinstance(t, Token)]

    @property
    def children(self) -> Sequence["Context"]:
        """Returns all child contexts."""
        return [c for c in self.body if isinstance(c, Context)]

    def as_string(self) -> str:
        """Returns a string representation of this context."""
        return "".join(t.token_str for t in self.tokens)

    def as_token_ids(self) -> list[int]:
        """Returns a list of token IDs stored in this context."""
        return [t.token_id for t in self.tokens]

    async def prompt(self, text: str, strip: bool = False) -> Sequence[Token]:
        """Tokenizes the argument and adds the tokens to the context.

        Args:
            text: String to tokenize and add to the context.
            strip: If true, any whitespace surrounding `prompt` will be stripped.

        Returns:
            Tokenized string.
        """
        if strip:
            text = text.strip()
        token_protos = await self._tokenize(text)

        request = {
            "contextId": self.context_id,
            "tokens": token_protos,
        }
        await js.pushTokens(pyodide.ffi.create_proxy(request))

        tokens = [Token.from_proto_dict(t) for t in token_protos]
        self.body.extend(tokens)
        return tokens

    def randomize_rng_seed(self) -> int:
        """Samples a new RNG seed and returns it."""
        self.next_rng_seed = random.randint(0, 100000)
        return self.next_rng_seed

    def create_context(self) -> "Context":
        """Creates a new context and adds it as child context."""
        child = Context(
            parent=self, next_rng_seed=self._get_next_rng_seed(), model_name=self.model_name
        )
        return child

    def _get_next_rng_seed(self) -> int:
        """Returns the next RNG seed."""
        self.next_rng_seed += 1
        return self.next_rng_seed - 1

    async def sample(
        self,
        max_len: int = 256,
        temperature: float = 0.7,
        nucleus_p: float = 0.95,
        stop_tokens: Optional[list[str]] = None,
        stop_strings: Optional[list[str]] = None,
        rng_seed: Optional[int] = None,
        add_to_context: bool = True,
        return_attention: bool = False,
        allowed_tokens: Optional[Sequence[Union[int, str]]] = None,
        disallowed_tokens: Optional[Sequence[Union[int, str]]] = None,
        augment_tokens: bool = True,
    ) -> SampleResult:
        """Generates a model response based on the current prompt.

        The current prompt consists of all text that has been added to the prompt either since the
        beginning of the program.

        Args:
            max_len: Maximum number of tokens to generate.
            temperature: Temperature of the final softmax operation. The lower the temperature, the
                lower the variance of the token distribution. In the limit, the distribution collapses
                onto the single token with the highest probability.
            nucleus_p: Threshold of the Top-P sampling technique: We rank all tokens by their
                probability and then only actually sample from the set of tokens that ranks in the
                Top-P percentile of the distribution.
            stop_tokens: A list of strings, each of which will be mapped independently to a single
                token. If a string does not map cleanly to one token, it will be silently ignored.
                If the network samples one of these tokens, sampling is stopped and the stop token
                *is not* included in the response.
            stop_strings: A list of strings. If any of these strings occurs in the network output,
                sampling is stopped but the string that triggered the stop *will be* included in the
                response. Note that the response may be longer than the stop string. For example, if
                the stop string is "Hel" and the network predicts the single-token response "Hello",
                sampling will be stopped but the response will still read "Hello".
            rng_seed: See of the random number generator used to sample from the model outputs.
            add_to_context: If true, the generated tokens will be added to the context.
            return_attention: If true, returns the attention mask. Note that this can significantly
                increase the response size for long sequences.
            allowed_tokens: If set, only these tokens can be sampled. Invalid input tokens are
                ignored. Only one of `allowed_tokens` and `disallowed_tokens` must be set.
            disallowed_tokens: If set, these tokens cannot be sampled. Invalid input tokens are
                ignored. Only one of `allowed_tokens` and `disallowed_tokens` must be set.
            augment_tokens: If true, strings passed to `stop_tokens`, `allowed_tokens` and
                `disallowed_tokens` will be augmented to include both the passed token and the
                version with leading whitespace. This is useful because most words have two
                corresponding vocabulary entries: one with leading whitespace and one without.

        Returns:
            The generated text.
        """
        if max_len is None and not stop_tokens:
            raise ValueError("Must provide either max_len or stop_tokens when calling `generate`.")

        if rng_seed is None:
            rng_seed = self._get_next_rng_seed()

        if max_len is not None:
            print(
                f"Generating {max_len} tokens [seed={rng_seed}, temperature={temperature}, "
                f"nucleus_p={nucleus_p}, stop_tokens={stop_tokens}, stop_strings={stop_strings}]."
            )

        if augment_tokens:
            if stop_tokens:
                stop_tokens = stop_tokens + [f"▁{t}" for t in stop_tokens]
            if allowed_tokens:
                allowed_tokens = list(allowed_tokens) + [
                    f"▁{t}" for t in allowed_tokens if isinstance(t, str) and not t.startswith("▁")
                ]
            if disallowed_tokens:
                disallowed_tokens = list(disallowed_tokens) + [
                    f"▁{t}"
                    for t in disallowed_tokens
                    if isinstance(t, str) and not t.startswith("▁")
                ]

        request = {
            "prompt": self.as_token_ids(),
            "settings": {
                "maxLen": max_len or 0,
                "temperature": temperature,
                "nucleusP": nucleus_p,
                "stopTokens": stop_tokens or [],
                "stopStrings": stop_strings or [],
                "rngSeed": rng_seed,
                "allowedTokens": [_parse_input_token(t) for t in allowed_tokens or []],
                "disallowedTokens": [_parse_input_token(t) for t in disallowed_tokens or []],
            },
            "returnAttention": return_attention,
            "modelName": self.model_name,
        }

        args = pyodide.ffi.create_proxy(request)
        iterator = js.generate(args)

        result = SampleResult(request)

        while True:
            obj = await iterator.next()
            if obj.done:
                break

            token_proto = obj.value.to_py()
            result.append(Token.from_proto_dict(token_proto))

            if add_to_context:
                self.body.append(result.tokens[-1])

            # Sync the token to the UI thread.
            request = {
                "contextId": self.context_id,
                "tokens": [token_proto],
            }
            await js.pushTokens(pyodide.ffi.create_proxy(request))

        result.print_progress()
        return result

    def clone(self) -> "Context":
        """Clones the current prompt."""
        # We can't use deepcopy here because we need to make sure the clone is correctly synced to
        # the UI thread.
        clone = Context(
            # We only clone the tokens, not the child contexts.
            body=list(self.tokens),
            parent=self,
            next_rng_seed=self.next_rng_seed,
        )
        self.body.append(clone)
        return clone

    async def set_title(self, title: str):
        """Sets the title of the context, which is shown in the UI."""
        request = {
            "contextId": self.context_id,
            "title": title,
        }
        await js.setContextTitle(pyodide.ffi.create_proxy(request))

    def __enter__(self):
        """Uses this context as the current context."""
        if self._reset_token is not None:
            raise RuntimeError("Cannot enter a context twice.")
        self._reset_token = _current_ctx.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the context and resets the global state."""
        _current_ctx.reset(self._reset_token)
        self._reset_token = None


# The current context, which is used by the free-functions below.
_current_ctx = contextvars.ContextVar("root_ctx", default=Context(context_id=""))
await _current_ctx.get().set_title("Main")  # noqa: F704

# If set, overrides the current context. Useful to ignore newly created contexts in `prompt_fn`.
_force_ctx = contextvars.ContextVar("force_ctx", default=None)


def get_context() -> Context:
    """Returns the current context."""
    if _force_ctx.get() is not None:
        return _force_ctx.get()
    return _current_ctx.get()


@contextlib.contextmanager
def force_context(ctx: Context):
    """Overrides the current context with the provided one."""
    token = _force_ctx.set(ctx)
    try:
        yield
    finally:
        _force_ctx.reset(token)


# The following functions operate on the current context.


def as_string() -> str:
    """See `Context.as_string`."""
    return get_context().as_string()


def select_model(model_name: str):
    """See `Context.select_model`."""
    return get_context().select_model(model_name)


def as_token_ids() -> list[int]:
    """See `Context.as_token_ids`."""
    return get_context().as_token_ids()


async def prompt(text: str, strip: bool = False) -> Sequence[Token]:
    """See `Context.prompt`."""
    return await get_context().prompt(text, strip)


def randomize_rng_seed() -> int:
    """See `Context.randomize_rng_seed`."""
    return get_context().randomize_rng_seed()


def create_context() -> "Context":
    """See `Context.create_context()`."""
    return get_context().create_context()


async def set_title(title: str):
    """See `Context.set_title`."""
    await get_context().set_title(title)


async def sample(
    max_len: int = 256,
    temperature: float = 1.0,
    nucleus_p: float = 0.7,
    stop_tokens: Optional[list[str]] = None,
    stop_strings: Optional[list[str]] = None,
    rng_seed: Optional[int] = None,
    add_to_context: bool = True,
    return_attention: bool = False,
    allowed_tokens: Optional[Sequence[Union[int, str]]] = None,
    disallowed_tokens: Optional[Sequence[Union[int, str]]] = None,
):
    """See `Context.sample`."""
    return await get_context().sample(
        max_len,
        temperature,
        nucleus_p,
        stop_tokens,
        stop_strings,
        rng_seed,
        add_to_context,
        return_attention,
        allowed_tokens,
        disallowed_tokens,
    )


def clone() -> "Context":
    """See `Context.clone`."""
    return get_context().clone()


def prompt_fn(fn):
    """A context manager that executes `fn` in a fresh prompt context.

    If a function is annotated with this context manager, a fresh prompt context is created that
    the function operates on. This allows solving sub-problems with different prompt and
    incorporating the solution to a sub problems into the original one.

    Example:
        ```
        @prompt_fn
        async def add(a, b):
            prompt(f"{a}+{b}=")
            result = await sample(max_len=10, stop_strings=[" "])
            return result.as_string().split(" ")[0]
        ```

    In order to get access to the context used by an annotated function, the function must return
    it like this:

    ```
        @prompt_fn
        def foo():
            return get_context()
    ```

    You can override the context an annotated function uses. This is useful if you want to continue
    operating on a context that was created by a function.

    ```
        @prompt_fn
        async def bar():
            async prompt("1+1=")
            return get_context()

        @prompt_fn
        async def foo():
            await sample(max_len=24)

        ctx = await bar()
        with force_context(ctx):
            foo()
    ```

    Args:
        fn: An asynchronous function to execute in a newly created context.

    Returns:
        The wrapped function.
    """

    async def _fn(*args, **kwargs):
        with get_context().create_context() as ctx:
            await ctx.set_title(fn.__name__)
            return await fn(*args, **kwargs)

    return _fn


async def read_file(file_name: str) -> bytes:
    """Reads a file that the user has uploaded to the file manager.

    Args:
        file_name: Name of the file to read.

    Returns:
        The file's content as raw bytes array.
    """
    result = await js.readFile(pyodide.ffi.create_proxy(file_name))
    return result.to_py().tobytes()


async def write_file(
    file_name: str,
    content: bytes,
    mime_type: str = "application/octet-stream",
    overwrite: bool = True,
):
    """Stores a file in the IDE.

    Args:
        file_name: Name of the file to write.
        content: File content as a byte array.
        mime_type: The MIME type of the file.
        overwrite: If the file already exists, overwrite it.
    """
    requests = {
        "fileName": file_name,
        "content": content,
        "mimeType": mime_type,
        "overwrite": overwrite,
    }
    await js.writeFile(pyodide.ffi.create_proxy(requests))


# New prompt.
# Find the full documentation including examples under https://developers.x.ai/python-sdk/ide/

await prompt("The answer to life and the universe is")
await sample(max_len=3, return_attention=True)


# End of user code.