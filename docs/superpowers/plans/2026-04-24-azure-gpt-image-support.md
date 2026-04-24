# Azure gpt-image Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `:image` operation support to the Azure provider so `ReqLLM.generate_image/3` works with `gpt-image-*` deployments, routing to `/images/generations` (text-to-image, JSON) or `/images/edits` (image-to-image, multipart) based on whether the context carries image ContentParts.

**Architecture:** New formatter module `ReqLLM.Providers.Azure.Images` owns body construction and response parsing, siblings to the existing `Azure.OpenAI` / `Azure.Anthropic` formatters. A new `prepare_request(:image, …)` clause in `ReqLLM.Providers.Azure` classifies the context, resolves deployment/api_version/base_url, dispatches to the formatter, and builds a `Req.Request`. A `:skip_content_type` private flag is threaded through `attach/3` so multipart requests keep Req's auto-generated `content-type` (with boundary) rather than the JSON default.

**Tech Stack:** Elixir, Req (HTTP), NimbleOptions (schema), ExUnit (tests), ReqLLM provider plumbing (`ReqLLM.Provider`, `ReqLLM.Context`, `ReqLLM.Message.ContentPart`, `ReqLLM.Response`).

---

## Spec reference

`docs/superpowers/specs/2026-04-24-azure-gpt-image-support-design.md`

## File structure

**Create:**
- `lib/req_llm/providers/azure/images.ex` — formatter for `:image` operation on Azure. Exposes `format_generate_request/3`, `format_edit_request/4`, `parse_response/3`, `media_type_for/1`, `validate_image_parts!/1`.

**Modify:**
- `lib/req_llm/providers/azure.ex`
  - Add `"gpt-image"` to `@model_families` map (value `__MODULE__.Images`).
  - Add `"gpt-image" => "AZURE_OPENAI_BASE_URL"` to `@family_env_vars`.
  - Add `"gpt-image" => "AZURE_OPENAI_API_KEY"` to `@family_api_key_env_vars`.
  - Add new `prepare_request(:image, …)` clause.
  - Add private helpers: `image_context/2`, `classify_image_op/1`, `validate_image_model/1`, `reject_unsupported_endpoint_format/1`, `image_endpoint_path/3`, `build_image_request/7`.
  - Modify `attach/3` to honor `:skip_content_type` private flag.
- `test/provider/azure/azure_test.exs`
  - New describe block `"prepare_request/4 :image"` covering generate routing, edit routing, mask handling, auth, error paths.
  - Small helper `gpt_image_model/0` to synthesize the model struct (mirrors `traditional_openai_model/0`).
- `guides/azure.md` — new "Image generation" subsection.
- `guides/image-generation.md` — add Azure entry alongside OpenAI/Google; remove "not yet supported" language if present.

## Dependencies between tasks

- Tasks 1–3 define model family + formatter primitives; they can land in order.
- Task 4 depends on 1–3 for the generate path.
- Tasks 5–8 add edit-path pieces; 5 and 6 are independent, 7 and 8 depend on 5–7.
- Task 9 (error paths) depends on 4 and 8 being in place.
- Task 10 (docs) depends on everything else being stable.

---

## Task 1: Register gpt-image family and stub formatter module

**Files:**
- Create: `lib/req_llm/providers/azure/images.ex`
- Modify: `lib/req_llm/providers/azure.ex` (model family maps)
- Test: `test/provider/azure/azure_test.exs` (family routing)

- [ ] **Step 1: Write the failing test**

Append to `test/provider/azure/azure_test.exs` just above `defp get_header/2`:

```elixir
  describe "prepare_request/4 :image (family routing)" do
    test "gpt-image model routes to Azure.Images formatter" do
      model = gpt_image_model()

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          "A red fox",
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "test-key"
        )

      assert Req.Request.get_private(request, :formatter) == ReqLLM.Providers.Azure.Images
    end
  end
```

Also add the helper at the bottom of the file, after `traditional_openai_model/0`:

```elixir
  defp gpt_image_model do
    %LLMDB.Model{
      id: "gpt-image-1.5",
      provider: :azure,
      capabilities: %{chat: false},
      extra: %{}
    }
  end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `mix test test/provider/azure/azure_test.exs --only line:<new-line> --trace`

Expected: a failure because `prepare_request(:image, ...)` falls through to the default and returns an error (no `:image` clause exists yet).

- [ ] **Step 3: Create the formatter module skeleton**

Create `lib/req_llm/providers/azure/images.ex`:

```elixir
defmodule ReqLLM.Providers.Azure.Images do
  @moduledoc """
  Image generation and edit formatter for the Azure provider.

  Covers Azure OpenAI Service deployments of gpt-image-* models:
    POST /openai/deployments/{deployment}/images/generations (JSON)
    POST /openai/deployments/{deployment}/images/edits       (multipart)

  Selection between the two endpoints is made by `ReqLLM.Providers.Azure`
  based on the content of the normalized context.
  """

  alias ReqLLM.Context
  alias ReqLLM.Message
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Response

  @doc "Build the JSON body for /images/generations."
  def format_generate_request(_model_id, prompt, _opts) when is_binary(prompt) do
    %{"prompt" => prompt, "n" => 1}
  end

  @doc "Build the multipart parts list for /images/edits."
  def format_edit_request(_model_id, prompt, image_parts, _opts) do
    {prompt, image_parts}
  end

  @doc "Parse an image response body into a ReqLLM.Response."
  def parse_response(_body, _model, _opts), do: {:error, :not_implemented}
end
```

Stubbing the three functions now keeps later tasks small; each gets filled in independently.

- [ ] **Step 4: Register the family in `lib/req_llm/providers/azure.ex`**

Edit the `@model_families` map (currently lines ~270–281) to insert `"gpt-image"`:

```elixir
@model_families %{
  "gpt" => __MODULE__.OpenAI,
  "gpt-image" => __MODULE__.Images,
  "text-embedding" => __MODULE__.OpenAI,
  "codex" => __MODULE__.OpenAI,
  "o1" => __MODULE__.OpenAI,
  "o3" => __MODULE__.OpenAI,
  "o4" => __MODULE__.OpenAI,
  "deepseek" => __MODULE__.OpenAI,
  "mai-ds" => __MODULE__.OpenAI,
  "claude" => __MODULE__.Anthropic,
  "grok" => __MODULE__.OpenAI
}
```

Add the `"gpt-image"` entry to both `@family_env_vars` (~line 289) and `@family_api_key_env_vars` (~line 301):

```elixir
@family_env_vars %{
  "claude" => "AZURE_ANTHROPIC_BASE_URL",
  "gpt" => "AZURE_OPENAI_BASE_URL",
  "gpt-image" => "AZURE_OPENAI_BASE_URL",
  "text-embedding" => "AZURE_OPENAI_BASE_URL",
  ...
}

@family_api_key_env_vars %{
  "claude" => "AZURE_ANTHROPIC_API_KEY",
  "gpt" => "AZURE_OPENAI_API_KEY",
  "gpt-image" => "AZURE_OPENAI_API_KEY",
  "text-embedding" => "AZURE_OPENAI_API_KEY",
  ...
}
```

Prefix ordering (`@model_family_prefixes`) is rebuilt automatically because it uses `Map.keys/1` + longest-first sort — `"gpt-image"` will correctly win over `"gpt"` for model ids starting with `gpt-image-`.

- [ ] **Step 5: Add a minimal `prepare_request(:image, …)` clause so the routing test can observe the formatter**

Add this clause to `lib/req_llm/providers/azure.ex` immediately before the catch-all `def prepare_request(operation, model_spec, input, opts)`:

```elixir
def prepare_request(:image, model_spec, prompt_or_messages, opts) do
  with {:ok, model} <- ReqLLM.model(model_spec),
       model_id = effective_model_id(model),
       :ok <- validate_image_model(model_id),
       {:ok, context, prompt, _image_parts} <- image_context(prompt_or_messages, opts) do
    model_family = get_model_family(model_id)
    resolved_base_url = resolve_base_url(model_family, opts)

    opts_with_context =
      opts
      |> Keyword.put(:context, context)
      |> Keyword.put(:base_url, resolved_base_url)

    {:ok, processed_opts} =
      ReqLLM.Provider.Options.process(__MODULE__, :image, model, opts_with_context)

    {api_version, deployment, base_url} =
      extract_azure_credentials(model, processed_opts)

    formatter = __MODULE__.Images

    body = formatter.format_generate_request(model_id, prompt, processed_opts)
    path = "/deployments/#{deployment}/images/generations?api-version=#{api_version}"

    http_opts = Keyword.get(opts, :req_http_options, [])
    req_keys = supported_provider_options() ++ @common_req_keys

    request =
      Req.new(
        [
          url: path,
          method: :post,
          json: body,
          receive_timeout: Keyword.get(processed_opts, :receive_timeout, 120_000)
        ] ++ http_opts
      )
      |> Req.Request.register_options(req_keys)
      |> Req.Request.merge_options(
        Keyword.take(processed_opts, req_keys) ++
          [operation: :image, model: model.id, base_url: base_url]
      )
      |> Req.Request.put_private(:model, model)
      |> Req.Request.put_private(:formatter, formatter)
      |> attach(model, processed_opts)

    {:ok, request}
  end
end
```

Add the helper definitions at the bottom of the module (just before the final `end`):

```elixir
defp validate_image_model(model_id) do
  case get_model_family(model_id) do
    "gpt-image" ->
      :ok

    family ->
      {:error,
       ReqLLM.Error.Invalid.Parameter.exception(
         parameter:
           "Model '#{model_id}' (family '#{family}') does not support image operations on Azure. Use a gpt-image-* deployment."
       )}
  end
end

defp image_context(prompt_or_messages, opts) do
  context_result =
    case Keyword.get(opts, :context) do
      %ReqLLM.Context{} = ctx -> {:ok, ctx}
      _ -> ReqLLM.Context.normalize(prompt_or_messages, opts)
    end

  with {:ok, context} <- context_result do
    {prompt, image_parts} = extract_prompt_and_images(context)
    {:ok, context, prompt, image_parts}
  end
end

defp extract_prompt_and_images(%ReqLLM.Context{messages: messages}) do
  last_user =
    messages
    |> Enum.reverse()
    |> Enum.find(&(&1.role == :user))

  case last_user do
    %ReqLLM.Message{content: content} when is_list(content) ->
      prompt =
        content
        |> Enum.filter(&(&1.type == :text))
        |> Enum.map_join("", & &1.text)
        |> String.trim()

      images = Enum.filter(content, &(&1.type == :image))
      {prompt, images}

    %ReqLLM.Message{content: content} when is_binary(content) ->
      {String.trim(content), []}

    _ ->
      {"", []}
  end
end
```

- [ ] **Step 6: Run the test and confirm it passes**

Run: `mix test test/provider/azure/azure_test.exs --only line:<new-line> --trace`

Expected: PASS.

Also run the whole test file to catch regressions:

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add lib/req_llm/providers/azure.ex lib/req_llm/providers/azure/images.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): register gpt-image model family and Images formatter skeleton"
```

---

## Task 2: `format_generate_request/3` JSON body

**Files:**
- Modify: `lib/req_llm/providers/azure/images.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Add inside the existing `describe "prepare_request/4 :image (family routing)"` block (or a new describe block `"Azure.Images body encoding"`):

```elixir
  describe "Azure.Images.format_generate_request/3" do
    alias ReqLLM.Providers.Azure.Images

    test "emits minimum required fields" do
      body = Images.format_generate_request("gpt-image-1.5", "hello", [])
      assert body == %{"prompt" => "hello", "n" => 1}
    end

    test "passes through size, quality, output_format and user" do
      body =
        Images.format_generate_request(
          "gpt-image-1.5",
          "hello",
          size: "1024x1024",
          quality: :medium,
          output_format: :png,
          user: "u1"
        )

      assert body["size"] == "1024x1024"
      assert body["quality"] == "medium"
      assert body["output_format"] == "png"
      assert body["user"] == "u1"
    end

    test "passes provider_options for output_compression / background / moderation" do
      body =
        Images.format_generate_request(
          "gpt-image-1.5",
          "hello",
          provider_options: [output_compression: 80, background: "transparent", moderation: "low"]
        )

      assert body["output_compression"] == 80
      assert body["background"] == "transparent"
      assert body["moderation"] == "low"
    end

    test "accepts a size tuple" do
      body = Images.format_generate_request("gpt-image-1.5", "hi", size: {1536, 1024})
      assert body["size"] == "1536x1024"
    end

    test "omits keys whose values are nil" do
      body = Images.format_generate_request("gpt-image-1.5", "hi", size: nil, quality: nil)
      refute Map.has_key?(body, "size")
      refute Map.has_key?(body, "quality")
    end
  end
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure.Images.format_generate_request"`
Expected: failures because the stub returns `%{"prompt" => prompt, "n" => 1}` regardless of opts.

- [ ] **Step 3: Implement `format_generate_request/3`**

Replace the stub in `lib/req_llm/providers/azure/images.ex` with the following. `provider_options` may arrive as either a keyword list or a map (the base schema allows both), so normalize it once at the top.

```elixir
def format_generate_request(_model_id, prompt, opts) when is_binary(prompt) do
  provider_opts = normalize_provider_opts(opts)

  %{"prompt" => prompt, "n" => Keyword.get(opts, :n, 1)}
  |> maybe_put_size(Keyword.get(opts, :size))
  |> maybe_put_string("quality", Keyword.get(opts, :quality))
  |> maybe_put_string("output_format", Keyword.get(opts, :output_format))
  |> maybe_put_integer("output_compression", Keyword.get(provider_opts, :output_compression))
  |> maybe_put_string("background", Keyword.get(provider_opts, :background))
  |> maybe_put_string("moderation", Keyword.get(provider_opts, :moderation))
  |> maybe_put_string("user", Keyword.get(opts, :user))
end

defp normalize_provider_opts(opts) do
  case Keyword.get(opts, :provider_options, []) do
    list when is_list(list) -> list
    map when is_map(map) -> Enum.into(map, [])
  end
end

defp maybe_put_size(body, nil), do: body

defp maybe_put_size(body, {w, h}) when is_integer(w) and is_integer(h) do
  Map.put(body, "size", "#{w}x#{h}")
end

defp maybe_put_size(body, size) when is_binary(size), do: Map.put(body, "size", size)

defp maybe_put_string(body, _key, nil), do: body

defp maybe_put_string(body, key, value) when is_atom(value),
  do: Map.put(body, key, Atom.to_string(value))

defp maybe_put_string(body, key, value) when is_binary(value), do: Map.put(body, key, value)

defp maybe_put_integer(body, _key, nil), do: body

defp maybe_put_integer(body, key, value) when is_integer(value), do: Map.put(body, key, value)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure.Images.format_generate_request"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure/images.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): encode images/generations body"
```

---

## Task 3: `parse_response/3` b64_json decoding

**Files:**
- Modify: `lib/req_llm/providers/azure/images.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Add another describe block:

```elixir
  describe "Azure.Images.parse_response/3" do
    alias ReqLLM.Message.ContentPart
    alias ReqLLM.Providers.Azure.Images

    test "decodes b64_json entries into :image ContentParts" do
      body = %{
        "created" => 1_700_000_000,
        "data" => [
          %{"b64_json" => Base.encode64("PNG-BYTES-1")},
          %{"b64_json" => Base.encode64("PNG-BYTES-2")}
        ]
      }

      model = %LLMDB.Model{id: "gpt-image-1.5", provider: :azure}

      {:ok, resp} = Images.parse_response(body, model, [])

      assert [
               %ContentPart{type: :image, data: "PNG-BYTES-1", media_type: "image/png"},
               %ContentPart{type: :image, data: "PNG-BYTES-2", media_type: "image/png"}
             ] = resp.message.content

      assert resp.model == "gpt-image-1.5"
      assert resp.finish_reason == :stop
    end

    test "honors output_format opt for media_type" do
      body = %{"data" => [%{"b64_json" => Base.encode64("X")}]}
      model = %LLMDB.Model{id: "gpt-image-1.5", provider: :azure}

      {:ok, resp} = Images.parse_response(body, model, output_format: :webp)

      assert [%ContentPart{media_type: "image/webp"}] = resp.message.content
    end

    test "handles url entries (no b64_json)" do
      body = %{"data" => [%{"url" => "https://example.com/out.png"}]}
      model = %LLMDB.Model{id: "gpt-image-1.5", provider: :azure}

      {:ok, resp} = Images.parse_response(body, model, [])

      assert [%ContentPart{type: :image_url, url: "https://example.com/out.png"}] =
               resp.message.content
    end
  end
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure.Images.parse_response"`
Expected: failures — stub returns `{:error, :not_implemented}`.

- [ ] **Step 3: Implement `parse_response/3`**

Replace the stub in `lib/req_llm/providers/azure/images.ex`:

```elixir
def parse_response(body, model, opts) when is_map(body) do
  data = Map.get(body, "data", [])
  media_type = media_type_for(Keyword.get(opts, :output_format))

  parts =
    data
    |> Enum.map(&decode_item(&1, media_type))
    |> Enum.reject(&is_nil/1)

  context = Keyword.get(opts, :context) || %Context{messages: []}

  response = %Response{
    id: image_response_id(),
    model: model.id,
    context: context,
    message: %Message{role: :assistant, content: parts},
    object: nil,
    stream?: false,
    stream: nil,
    usage: nil,
    finish_reason: :stop,
    provider_meta: %{"azure" => Map.delete(body, "data")},
    error: nil
  }

  {:ok, Context.merge_response(context, response)}
end

@doc false
def media_type_for(:jpeg), do: "image/jpeg"
def media_type_for("jpeg"), do: "image/jpeg"
def media_type_for(:webp), do: "image/webp"
def media_type_for("webp"), do: "image/webp"
def media_type_for(_), do: "image/png"

defp decode_item(%{"b64_json" => b64} = item, media_type) when is_binary(b64) do
  revised = Map.get(item, "revised_prompt")
  metadata = if is_binary(revised), do: %{revised_prompt: revised}, else: %{}

  %ContentPart{
    type: :image,
    data: Base.decode64!(b64),
    media_type: media_type,
    metadata: metadata
  }
end

defp decode_item(%{"url" => url} = item, _media_type) when is_binary(url) do
  revised = Map.get(item, "revised_prompt")
  metadata = if is_binary(revised), do: %{revised_prompt: revised}, else: %{}
  %ContentPart{type: :image_url, url: url, metadata: metadata}
end

defp decode_item(_, _), do: nil

defp image_response_id do
  "img_" <> (:crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false))
end
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure.Images.parse_response"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure/images.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): decode images response into ReqLLM.Response"
```

---

## Task 4: End-to-end request shape for the generate path

**Files:**
- Modify: `lib/req_llm/providers/azure.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Append to the `"prepare_request/4 :image (family routing)"` describe block:

```elixir
    test "builds /images/generations request with JSON body" do
      model = gpt_image_model()

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          "A red fox",
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "test-key",
          size: "1024x1024",
          quality: :medium,
          output_format: :png,
          provider_options: [output_compression: 100]
        )

      url_string = URI.to_string(request.url)
      assert url_string =~ "/deployments/gpt-image-1.5/images/generations"
      assert url_string =~ "api-version="

      assert request.method == :post

      body = get_json_body(request)
      assert body["prompt"] == "A red fox"
      assert body["size"] == "1024x1024"
      assert body["quality"] == "medium"
      assert body["output_format"] == "png"
      assert body["output_compression"] == 100

      refute Map.has_key?(body, "model"),
             "traditional Azure endpoint must not carry model in body"
    end

    test "api-key auth header is set" do
      model = gpt_image_model()

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          "hi",
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "secret"
        )

      assert get_header(request.headers, "api-key") == "secret"
    end

    test "bearer token auth header is set when api_key starts with Bearer" do
      model = gpt_image_model()

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          "hi",
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "Bearer token-abc"
        )

      assert get_header(request.headers, "authorization") == "Bearer token-abc"
      assert get_header(request.headers, "api-key") == nil
    end
```

- [ ] **Step 2: Run the tests and confirm they fail or partially fail**

Run: `mix test test/provider/azure/azure_test.exs -k "prepare_request/4 :image"`
Expected: PASS for those tests that use Task 1's stub (formatter private, basic routing), FAIL for the new JSON body / auth tests because the JSON body is sparse (Task 2's format_generate_request already landed but options may not have flowed through Options.process — verify in the next step).

If all four tests already pass after Tasks 1–3, skip to Step 4.

- [ ] **Step 3: Verify option flow through `Options.process` and repair if needed**

If any body assertion fails, inspect the body:

```elixir
{:ok, request} = Azure.prepare_request(:image, gpt_image_model(), "hi", [...])
IO.inspect(request.options[:json])
```

Likely cause: `ReqLLM.Provider.Options.process(__MODULE__, :image, model, opts)` may strip unknown keys that aren't in Azure's `@provider_schema`. If `:output_compression` etc. don't make it through:

1. Confirm: check whether `Keyword.get(processed_opts, :provider_options, [])` still contains the expected keys.
2. Fix: in `prepare_request(:image, …)`, fall back to the pre-processed `opts` when reading image-specific fields, or explicitly re-merge:

```elixir
processed_opts =
  Keyword.merge(processed_opts, Keyword.take(opts, [:size, :quality, :output_format, :user, :n]))
```

Add `:size`, `:quality`, `:output_format`, `:user`, `:n` to `req_keys` as well so they propagate to `Req.Request.merge_options`:

```elixir
req_keys =
  supported_provider_options() ++
    @common_req_keys ++
    [:size, :quality, :output_format, :user, :n]
```

- [ ] **Step 4: Run tests**

Run: `mix test test/provider/azure/azure_test.exs -k "prepare_request/4 :image"`
Expected: PASS.

Run the full file:

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): wire generate path to /images/generations"
```

---

## Task 5: `classify_image_op/1` + edit-side helpers

**Files:**
- Modify: `lib/req_llm/providers/azure.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Add a new describe block:

```elixir
  describe "Azure image op classification" do
    alias ReqLLM.Message.ContentPart

    test "text-only context routes to generate path" do
      model = gpt_image_model()

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          "a red fox",
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "k"
        )

      assert URI.to_string(request.url) =~ "/images/generations"
    end

    test "context with an image ContentPart routes to edit path" do
      model = gpt_image_model()

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user([
            ContentPart.text("Make it black and white"),
            ContentPart.image("IMG-BYTES", "image/png")
          ])
        ])

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          context,
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "k"
        )

      assert URI.to_string(request.url) =~ "/images/edits"
    end
  end
```

The second test will fail — the edit path isn't wired yet (Task 8). The first should already pass.

- [ ] **Step 2: Run and confirm the edit-routing test fails**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure image op classification"`
Expected: "text-only" test PASSES; "with image ContentPart" test FAILS (URL is `/images/generations` instead of `/images/edits`).

- [ ] **Step 3: Change `prepare_request(:image, …)` to branch on image_parts**

Modify the clause added in Task 1 so it dispatches on the `image_parts` return of `image_context/2`:

```elixir
def prepare_request(:image, model_spec, prompt_or_messages, opts) do
  with {:ok, model} <- ReqLLM.model(model_spec),
       model_id = effective_model_id(model),
       :ok <- validate_image_model(model_id),
       {:ok, context, prompt, image_parts} <- image_context(prompt_or_messages, opts) do
    sub_op = if image_parts == [], do: :generate, else: :edit

    model_family = get_model_family(model_id)
    resolved_base_url = resolve_base_url(model_family, opts)

    opts_with_context =
      opts
      |> Keyword.put(:context, context)
      |> Keyword.put(:base_url, resolved_base_url)

    {:ok, processed_opts} =
      ReqLLM.Provider.Options.process(__MODULE__, :image, model, opts_with_context)

    {api_version, deployment, base_url} =
      extract_azure_credentials(model, processed_opts)

    build_image_request(
      sub_op,
      model,
      prompt,
      image_parts,
      processed_opts,
      deployment,
      api_version,
      base_url,
      opts
    )
  end
end
```

Move the existing Req-building code into `build_image_request(:generate, model, prompt, _image_parts, processed_opts, deployment, api_version, base_url, opts)`:

```elixir
defp build_image_request(:generate, model, prompt, _image_parts, processed_opts, deployment, api_version, base_url, opts) do
  formatter = __MODULE__.Images
  body = formatter.format_generate_request(model.id, prompt, processed_opts)
  path = "/deployments/#{deployment}/images/generations?api-version=#{api_version}"

  http_opts = Keyword.get(opts, :req_http_options, [])
  req_keys = supported_provider_options() ++ @common_req_keys ++ [:size, :quality, :output_format, :user, :n]

  request =
    Req.new(
      [
        url: path,
        method: :post,
        json: body,
        receive_timeout: Keyword.get(processed_opts, :receive_timeout, 120_000)
      ] ++ http_opts
    )
    |> Req.Request.register_options(req_keys)
    |> Req.Request.merge_options(
      Keyword.take(processed_opts, req_keys) ++
        [operation: :image, model: model.id, base_url: base_url]
    )
    |> Req.Request.put_private(:model, model)
    |> Req.Request.put_private(:formatter, formatter)
    |> attach(model, processed_opts)

  {:ok, request}
end

defp build_image_request(:edit, model, _prompt, _image_parts, _processed_opts, deployment, api_version, base_url, _opts) do
  # Placeholder until Task 8 — returns a minimal request that routes to /edits so
  # the routing test in Task 5 passes. Task 8 will flesh this out with multipart.
  path = "/deployments/#{deployment}/images/edits?api-version=#{api_version}"

  request =
    Req.new(url: path, method: :post, base_url: base_url)
    |> Req.Request.register_options([:model, :base_url, :operation])
    |> Req.Request.merge_options(model: model.id, base_url: base_url, operation: :image)
    |> Req.Request.put_private(:model, model)
    |> Req.Request.put_private(:formatter, __MODULE__.Images)

  {:ok, request}
end
```

Leaving the `:edit` branch as a thin placeholder is intentional — Task 8 builds it out properly. The placeholder is enough to make Task 5's routing test green without tying Task 5 and Task 8 together.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure image op classification"`
Expected: both tests PASS.

Run the full file to make sure nothing else broke:

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): classify image op by context and route to /edits or /generations"
```

---

## Task 6: `format_edit_request/4` multipart parts

**Files:**
- Modify: `lib/req_llm/providers/azure/images.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Add another describe block:

```elixir
  describe "Azure.Images.format_edit_request/4" do
    alias ReqLLM.Message.ContentPart
    alias ReqLLM.Providers.Azure.Images

    test "emits image + prompt parts with correct content types" do
      parts = [ContentPart.image("IMG-BYTES", "image/png")]

      form = Images.format_edit_request("gpt-image-1.5", "make it b/w", parts, [])

      image_entry = Enum.find(form, fn {k, _} -> k == :image end)
      {:image, {bytes, meta}} = image_entry
      assert bytes == "IMG-BYTES"
      assert Keyword.fetch!(meta, :content_type) == "image/png"
      assert Keyword.fetch!(meta, :filename) == "image.png"

      assert {:prompt, "make it b/w"} in form
    end

    test "emits multiple image parts in order" do
      parts = [
        ContentPart.image("A", "image/png"),
        ContentPart.image("B", "image/jpeg")
      ]

      form = Images.format_edit_request("gpt-image-1.5", "compose", parts, [])
      images = for {:image, entry} <- form, do: entry

      assert length(images) == 2
      {bytes1, meta1} = Enum.at(images, 0)
      {bytes2, meta2} = Enum.at(images, 1)
      assert bytes1 == "A"
      assert bytes2 == "B"
      assert Keyword.fetch!(meta1, :content_type) == "image/png"
      assert Keyword.fetch!(meta2, :content_type) == "image/jpeg"
      assert Keyword.fetch!(meta2, :filename) == "image.jpg"
    end

    test "inserts mask when provided as binary (defaults to png)" do
      parts = [ContentPart.image("IMG", "image/png")]

      form =
        Images.format_edit_request(
          "gpt-image-1.5",
          "edit",
          parts,
          provider_options: [mask: "MASK-BYTES"]
        )

      {:mask, {bytes, meta}} = Enum.find(form, &match?({:mask, _}, &1))
      assert bytes == "MASK-BYTES"
      assert Keyword.fetch!(meta, :content_type) == "image/png"
      assert Keyword.fetch!(meta, :filename) == "mask.png"
    end

    test "inserts mask when provided as %ContentPart{type: :image}" do
      parts = [ContentPart.image("IMG", "image/png")]
      mask_cp = ContentPart.image("MASK", "image/webp")

      form =
        Images.format_edit_request(
          "gpt-image-1.5",
          "edit",
          parts,
          provider_options: [mask: mask_cp]
        )

      {:mask, {bytes, meta}} = Enum.find(form, &match?({:mask, _}, &1))
      assert bytes == "MASK"
      assert Keyword.fetch!(meta, :content_type) == "image/webp"
      assert Keyword.fetch!(meta, :filename) == "mask.webp"
    end

    test "passes scalar opts as string multipart entries" do
      parts = [ContentPart.image("IMG", "image/png")]

      form =
        Images.format_edit_request(
          "gpt-image-1.5",
          "edit",
          parts,
          n: 2,
          size: "1024x1024",
          quality: :high
        )

      assert {:n, "2"} in form
      assert {:size, "1024x1024"} in form
      assert {:quality, "high"} in form
    end

    test "raises on ContentPart with type :image_url (URL only)" do
      parts = [%ContentPart{type: :image_url, url: "https://x/y.png"}]

      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Images.format_edit_request("gpt-image-1.5", "edit", parts, [])
      end
    end

    test "raises when image_parts is empty" do
      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Images.format_edit_request("gpt-image-1.5", "edit", [], [])
      end
    end
  end
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure.Images.format_edit_request"`
Expected: failures — stub returns a 2-tuple of `{prompt, image_parts}`, not a multipart list.

- [ ] **Step 3: Implement `format_edit_request/4`**

Replace the stub in `lib/req_llm/providers/azure/images.ex`:

```elixir
def format_edit_request(_model_id, prompt, image_parts, opts) when is_binary(prompt) do
  :ok = validate_image_parts!(image_parts)

  provider_opts = normalize_provider_opts(opts)

  image_fields =
    Enum.map(image_parts, fn %ContentPart{data: bytes, media_type: mt} ->
      mt = mt || "image/png"
      {:image, {bytes, filename: "image.#{ext_for(mt)}", content_type: mt}}
    end)

  mask_field = build_mask_field(Keyword.get(provider_opts, :mask))

  image_fields ++
    mask_field ++
    [{:prompt, prompt}] ++
    maybe_multipart(:n, Keyword.get(opts, :n)) ++
    maybe_multipart(:size, Keyword.get(opts, :size)) ++
    maybe_multipart(:quality, Keyword.get(opts, :quality))
end

defp validate_image_parts!([]) do
  raise ReqLLM.Error.Invalid.Parameter.exception(
          parameter:
            "image edit requires at least one %ContentPart{type: :image} in the last user message"
        )
end

defp validate_image_parts!(parts) when is_list(parts) do
  Enum.each(parts, fn
    %ContentPart{type: :image, data: data} when is_binary(data) ->
      :ok

    %ContentPart{type: type} ->
      raise ReqLLM.Error.Invalid.Parameter.exception(
              parameter:
                "Azure image edit requires binary image data. Got ContentPart of type #{inspect(type)}; use ContentPart.image/2 with binary bytes."
            )
  end)

  :ok
end

defp build_mask_field(nil), do: []

defp build_mask_field(bytes) when is_binary(bytes),
  do: [{:mask, {bytes, filename: "mask.png", content_type: "image/png"}}]

defp build_mask_field(%ContentPart{type: :image, data: bytes, media_type: mt}) do
  mt = mt || "image/png"
  [{:mask, {bytes, filename: "mask.#{ext_for(mt)}", content_type: mt}}]
end

defp build_mask_field(other) do
  raise ReqLLM.Error.Invalid.Parameter.exception(
          parameter:
            "provider_options[:mask] must be binary bytes or a %ContentPart{type: :image}; got: #{inspect(other)}"
        )
end

defp ext_for("image/png"), do: "png"
defp ext_for("image/jpeg"), do: "jpg"
defp ext_for("image/webp"), do: "webp"
defp ext_for(_), do: "png"

defp maybe_multipart(_key, nil), do: []

defp maybe_multipart(key, value) when is_atom(value),
  do: [{key, Atom.to_string(value)}]

defp maybe_multipart(key, value) when is_integer(value),
  do: [{key, Integer.to_string(value)}]

defp maybe_multipart(key, {w, h}) when is_integer(w) and is_integer(h),
  do: [{key, "#{w}x#{h}"}]

defp maybe_multipart(key, value) when is_binary(value), do: [{key, value}]
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure.Images.format_edit_request"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure/images.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): build multipart body for /images/edits"
```

---

## Task 7: `:skip_content_type` flag in `attach/3`

**Files:**
- Modify: `lib/req_llm/providers/azure.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Add a small describe block:

```elixir
  describe "Azure.attach content-type handling" do
    test "JSON requests keep content-type application/json" do
      model = gpt_image_model()

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          "hi",
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "k"
        )

      assert get_header(request.headers, "content-type") =~ "application/json"
    end

    test "multipart image-edit requests do not carry a JSON content-type" do
      model = gpt_image_model()

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user([
            ReqLLM.Message.ContentPart.text("edit"),
            ReqLLM.Message.ContentPart.image("IMG", "image/png")
          ])
        ])

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          context,
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "k"
        )

      assert get_header(request.headers, "content-type") == nil
    end
  end
```

The multipart test will fail after Task 8 wires up the edit path properly, because `attach/3` currently always sets `content-type: application/json`. But with Task 7's change it stops setting it when the `:skip_content_type` private flag is present.

- [ ] **Step 2: Change `attach/3` to honor the flag**

Locate `attach/3` (around line 527 in `lib/req_llm/providers/azure.ex`). It contains this exact pipeline step:

```elixir
|> Req.Request.put_header("content-type", "application/json")
```

Replace that single line with:

```elixir
|> maybe_put_json_content_type()
```

leaving the rest of the pipeline (`|> Req.Request.put_header(auth_header_name, auth_header_value)`, the `then/2` header loop, etc.) untouched. Add the helper near the other private helpers, e.g. immediately after `reject_unsupported_endpoint_format/1` once Task 9 has added it, or right before the `defp extract_azure_credentials` block if Task 7 lands first:

```elixir
defp maybe_put_json_content_type(request) do
  if Req.Request.get_private(request, :skip_content_type) do
    request
  else
    Req.Request.put_header(request, "content-type", "application/json")
  end
end
```

This is a no-op for every existing code path (flag defaults to `nil`/`false`). Task 8 will set the flag for the edit request.

- [ ] **Step 3: Run all Azure tests and confirm no regressions**

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all previously-passing tests still pass. The "multipart image-edit requests do not carry a JSON content-type" test will still fail — that is intentional; Task 8 makes it pass.

- [ ] **Step 4: Commit**

```bash
git add lib/req_llm/providers/azure.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): honor :skip_content_type private flag in attach/3"
```

---

## Task 8: Wire the edit path into `build_image_request/9`

**Files:**
- Modify: `lib/req_llm/providers/azure.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Append inside the `"Azure image op classification"` describe block (added in Task 5):

```elixir
    test "edit request uses form_multipart with image + prompt + mask parts" do
      model = gpt_image_model()

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user([
            ReqLLM.Message.ContentPart.text("Make this black and white"),
            ReqLLM.Message.ContentPart.image("IMG-BYTES", "image/png")
          ])
        ])

      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          context,
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "k",
          provider_options: [mask: "MASK-BYTES"]
        )

      form = request.options[:form_multipart]
      refute is_nil(form), "expected form_multipart option to be set"

      assert Enum.find(form, &match?({:image, _}, &1))
      assert Enum.find(form, &match?({:mask, _}, &1))
      assert {:prompt, "Make this black and white"} in form

      # URL
      assert URI.to_string(request.url) =~ "/deployments/gpt-image-1.5/images/edits"

      # Auth header still set
      assert get_header(request.headers, "api-key") == "k"

      # Content-type not pre-set — Req will supply multipart boundary
      assert get_header(request.headers, "content-type") == nil
    end
```

- [ ] **Step 2: Run and confirm failure**

Run: `mix test test/provider/azure/azure_test.exs -k "edit request uses form_multipart"`
Expected: FAIL because `build_image_request(:edit, …)` is still the Task-5 placeholder.

- [ ] **Step 3: Replace the `:edit` branch of `build_image_request/9`**

Replace the placeholder with the real implementation:

```elixir
defp build_image_request(:edit, model, prompt, image_parts, processed_opts, deployment, api_version, base_url, opts) do
  formatter = __MODULE__.Images

  form_parts =
    formatter.format_edit_request(model.id, prompt, image_parts, processed_opts)

  path = "/deployments/#{deployment}/images/edits?api-version=#{api_version}"

  http_opts = Keyword.get(opts, :req_http_options, [])
  req_keys = supported_provider_options() ++ @common_req_keys ++ [:size, :quality, :output_format, :n]

  request =
    Req.new(
      [
        url: path,
        method: :post,
        form_multipart: form_parts,
        receive_timeout: Keyword.get(processed_opts, :receive_timeout, 120_000)
      ] ++ http_opts
    )
    |> Req.Request.register_options(req_keys ++ [:form_multipart])
    |> Req.Request.merge_options(
      Keyword.take(processed_opts, req_keys) ++
        [operation: :image, model: model.id, base_url: base_url]
    )
    |> Req.Request.put_private(:model, model)
    |> Req.Request.put_private(:formatter, formatter)
    |> Req.Request.put_private(:skip_content_type, true)
    |> attach(model, processed_opts)

  {:ok, request}
end
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `mix test test/provider/azure/azure_test.exs -k "edit request uses form_multipart"`
Expected: PASS.

Also verify the content-type test from Task 7 now passes:

Run: `mix test test/provider/azure/azure_test.exs -k "multipart image-edit requests do not carry a JSON content-type"`
Expected: PASS.

Full sweep:

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): implement /images/edits multipart request builder"
```

---

## Task 9: Error paths

**Files:**
- Modify: `lib/req_llm/providers/azure.ex`
- Test: `test/provider/azure/azure_test.exs`

- [ ] **Step 1: Write the failing tests**

Add a new describe block:

```elixir
  describe "Azure image operation error paths" do
    test "rejects :image for non-gpt-image model family" do
      model = traditional_openai_model()

      assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: msg}} =
               Azure.prepare_request(
                 :image,
                 model,
                 "hi",
                 base_url: "https://r.openai.azure.com/openai",
                 api_key: "k",
                 deployment: "d"
               )

      assert msg =~ "does not support image operations on Azure"
    end

    test "rejects Foundry base URL for :image" do
      model = gpt_image_model()

      assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: msg}} =
               Azure.prepare_request(
                 :image,
                 model,
                 "hi",
                 base_url: "https://r.services.ai.azure.com",
                 api_key: "k",
                 deployment: "d"
               )

      assert msg =~ "traditional Azure OpenAI Service"
    end

    test "rejects v1 GA base URL for :image" do
      model = gpt_image_model()

      assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: msg}} =
               Azure.prepare_request(
                 :image,
                 model,
                 "hi",
                 base_url: "https://r.openai.azure.com/openai/v1",
                 api_key: "k",
                 deployment: "d"
               )

      assert msg =~ "traditional Azure OpenAI Service"
    end

    test "rejects image edit with only an :image_url ContentPart" do
      model = gpt_image_model()

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user([
            ReqLLM.Message.ContentPart.text("edit"),
            %ReqLLM.Message.ContentPart{type: :image_url, url: "https://x/y.png"}
          ])
        ])

      # image_url does not count as :image — this context has zero image parts
      # and therefore routes to generate, which is fine. The specific "binary
      # required for edit" assertion is covered in Task 6. This test documents
      # the routing behavior.
      {:ok, request} =
        Azure.prepare_request(
          :image,
          model,
          context,
          deployment: "gpt-image-1.5",
          base_url: "https://r.openai.azure.com/openai",
          api_key: "k"
        )

      assert URI.to_string(request.url) =~ "/images/generations"
    end
  end
```

- [ ] **Step 2: Run and confirm failures**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure image operation error paths"`
Expected: the "non-gpt-image model family" test passes (Task 1 already added `validate_image_model/1`); the Foundry / v1 GA tests FAIL because `reject_unsupported_endpoint_format/1` isn't wired in yet.

- [ ] **Step 3: Add the endpoint-format guard**

Replace the full `prepare_request(:image, …)` clause from Task 5 with the version below. The only net change is the addition of three `with` bindings (`model_family = …`, `resolved_base_url = …`, `:ok <- reject_unsupported_endpoint_format(resolved_base_url)`); the body after `do` stays the same but now reads `resolved_base_url` from the `with` scope instead of recomputing it.

```elixir
def prepare_request(:image, model_spec, prompt_or_messages, opts) do
  with {:ok, model} <- ReqLLM.model(model_spec),
       model_id = effective_model_id(model),
       :ok <- validate_image_model(model_id),
       {:ok, context, prompt, image_parts} <- image_context(prompt_or_messages, opts),
       model_family = get_model_family(model_id),
       resolved_base_url = resolve_base_url(model_family, opts),
       :ok <- reject_unsupported_endpoint_format(resolved_base_url) do
    sub_op = if image_parts == [], do: :generate, else: :edit

    opts_with_context =
      opts
      |> Keyword.put(:context, context)
      |> Keyword.put(:base_url, resolved_base_url)

    {:ok, processed_opts} =
      ReqLLM.Provider.Options.process(__MODULE__, :image, model, opts_with_context)

    {api_version, deployment, base_url} =
      extract_azure_credentials(model, processed_opts)

    build_image_request(
      sub_op,
      model,
      prompt,
      image_parts,
      processed_opts,
      deployment,
      api_version,
      base_url,
      opts
    )
  end
end
```

Add the helper:

```elixir
defp reject_unsupported_endpoint_format(base_url) when is_binary(base_url) do
  cond do
    uses_foundry_format?(base_url) ->
      {:error,
       ReqLLM.Error.Invalid.Parameter.exception(
         parameter:
           "Image operations require the traditional Azure OpenAI Service base URL (https://<resource>.openai.azure.com/openai). Azure AI Foundry image endpoints are not supported."
       )}

    uses_v1_ga_format?(base_url) ->
      {:error,
       ReqLLM.Error.Invalid.Parameter.exception(
         parameter:
           "Image operations require the traditional Azure OpenAI Service base URL (https://<resource>.openai.azure.com/openai). The v1 GA path (/openai/v1) is not supported for images."
       )}

    true ->
      :ok
  end
end

defp reject_unsupported_endpoint_format(_), do: :ok
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `mix test test/provider/azure/azure_test.exs -k "Azure image operation error paths"`
Expected: PASS.

Full sweep:

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add lib/req_llm/providers/azure.ex test/provider/azure/azure_test.exs
git commit -m "feat(azure): reject Foundry/v1-GA base URLs for image ops"
```

---

## Task 10: Documentation

**Files:**
- Modify: `guides/azure.md`
- Modify: `guides/image-generation.md`

- [ ] **Step 1: Read the current state of `guides/azure.md`**

Read `guides/azure.md` to find the best place to insert an "Image generation" section — typically after "Embeddings" and before "Error handling" or similar.

- [ ] **Step 2: Append "Image generation" section to `guides/azure.md`**

Add this section, adapting the heading level to the surrounding document (two-hash-level example below):

````markdown
## Image generation

Azure deployments of `gpt-image-*` models support both text-to-image generation
(`/images/generations`) and image-to-image editing (`/images/edits`). Both are
reached through `ReqLLM.generate_image/3`; the request is routed to the edit
endpoint when the context contains `%ReqLLM.Message.ContentPart{type: :image}`
entries, and to the generate endpoint otherwise.

**Requirements:** Azure OpenAI Service (traditional) base URL — the form
`https://<resource>.openai.azure.com/openai`. Azure AI Foundry and the
`/openai/v1` GA path are not supported for image operations.

### Text-to-image (generations)

```elixir
{:ok, response} =
  ReqLLM.generate_image(
    "azure:gpt-image-1.5",
    "A photograph of a red fox in an autumn forest",
    base_url: "https://<resource>.openai.azure.com/openai",
    deployment: "gpt-image-1.5",
    size: "1024x1024",
    quality: :medium,
    output_format: :png,
    provider_options: [output_compression: 100]
  )

File.write!("fox.png", ReqLLM.Response.image_data(response))
```

### Image-to-image (edits)

```elixir
input = File.read!("image_to_edit.png")
mask  = File.read!("mask.png")

context =
  ReqLLM.Context.new([
    ReqLLM.Context.user([
      ReqLLM.Message.ContentPart.text("Make this black and white"),
      ReqLLM.Message.ContentPart.image(input, "image/png")
    ])
  ])

{:ok, response} =
  ReqLLM.generate_image(
    "azure:gpt-image-1.5",
    context,
    base_url: "https://<resource>.openai.azure.com/openai",
    deployment: "gpt-image-1.5",
    provider_options: [mask: mask]
  )

File.write!("edited.png", ReqLLM.Response.image_data(response))
```

The mask is optional. Multiple `ContentPart.image/2` parts in the same user
message are sent as multiple `image` multipart fields (used by gpt-image models
for composition).
````

- [ ] **Step 3: Update `guides/image-generation.md`**

Locate the OpenAI "Current Limitations" block at `guides/image-generation.md:85-91`. If it still claims image editing is unsupported, leave the OpenAI wording alone (it remains true for the openai provider in this scope) but add an Azure section after the Google section. Use the existing OpenAI section as a template:

````markdown
---

## Azure

Azure OpenAI Service hosts `gpt-image-*` deployments. Both text-to-image and
image-to-image are supported via `ReqLLM.generate_image/3` — the input context
selects the endpoint.

### Supported models

| Deployment model | Notes |
|------------------|-------|
| `gpt-image-1.5`  | State-of-the-art; supports generations and edits |
| `gpt-image-1`    | Generations and edits |
| `gpt-image-1-mini` | Generations |

### Constraint

Only the traditional Azure OpenAI Service base URL
(`https://<resource>.openai.azure.com/openai`) is supported for image
operations. Azure AI Foundry and the `/openai/v1` GA path return an error.

### Text-to-image

```elixir
{:ok, response} =
  ReqLLM.generate_image(
    "azure:gpt-image-1.5",
    "A photograph of a red fox in an autumn forest",
    base_url: "https://<resource>.openai.azure.com/openai",
    deployment: "gpt-image-1.5",
    size: "1024x1024",
    quality: :medium
  )
```

### Image-to-image (edits)

```elixir
context =
  ReqLLM.Context.new([
    ReqLLM.Context.user([
      ReqLLM.Message.ContentPart.text("Make this black and white"),
      ReqLLM.Message.ContentPart.image(File.read!("input.png"), "image/png")
    ])
  ])

{:ok, response} =
  ReqLLM.generate_image(
    "azure:gpt-image-1.5",
    context,
    base_url: "https://<resource>.openai.azure.com/openai",
    deployment: "gpt-image-1.5",
    provider_options: [mask: File.read!("mask.png")]
  )
```
````

- [ ] **Step 4: Commit**

```bash
git add guides/azure.md guides/image-generation.md
git commit -m "docs(azure): document gpt-image generations and edits"
```

---

## Final sweep

- [ ] **Step 1: Run all Azure tests**

Run: `mix test test/provider/azure/azure_test.exs`
Expected: all green.

- [ ] **Step 2: Run the full Azure provider module's compilation / type check**

Run: `mix compile --warnings-as-errors`
Expected: no warnings; no errors.

- [ ] **Step 3: Run the full library test suite to detect cross-cutting regressions**

Run: `mix test --exclude live`
Expected: all green. (Live-coverage tests are excluded — this change isn't verified against a real Azure endpoint as part of automated tests; see the smoke-test snippet in `guides/azure.md`.)

- [ ] **Step 4: Final commit / clean-up**

If any doc or comment fix falls out of the sweep, land it as a single follow-up commit:

```bash
git add -A
git commit -m "chore(azure): tidy after image-support implementation"
```

(Skip if nothing fell out.)
