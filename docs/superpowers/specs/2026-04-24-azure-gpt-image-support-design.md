# Azure provider: gpt-image generations and edits

Status: approved design
Date: 2026-04-24
Scope: `ReqLLM.Providers.Azure` — add image generation (`/images/generations`) and image-to-image editing (`/images/edits`) for `gpt-image-*` deployments on Azure OpenAI Service (traditional endpoint format only).

## 1. Motivation

The Azure provider currently routes only `:chat`, `:object`, and `:embedding` operations. Calling `ReqLLM.generate_image("azure:gpt-image-1.5", …)` falls through to `ReqLLM.Provider.Defaults.prepare_request/5`, which returns "operation not supported".

Users with Azure deployments of `gpt-image-1.5` (and related `gpt-image-*` models) cannot use them through ReqLLM. Two Azure endpoints are in scope:

- `POST /openai/deployments/{deployment}/images/generations?api-version=…` — JSON body, text-to-image.
- `POST /openai/deployments/{deployment}/images/edits?api-version=…` — multipart/form-data body, image-to-image with optional mask.

Both should be reachable through the existing single entry point `ReqLLM.generate_image/3`, matching the pattern Google already uses (image ContentParts in context → edit path, text-only → generate path).

## 2. Non-goals

- OpenAI provider `/images/edits` support. Out of scope for this change; can mirror this work in a follow-up.
- Azure AI Foundry (`.services.ai.azure.com`) and Azure OpenAI v1 GA (`/openai/v1`) endpoint formats for images. These produce an explicit error when used with `:image` on Azure — traditional Azure OpenAI Service format only.
- Streaming image responses (not offered by Azure image endpoints).
- Image variations (DALL-E 2 only; no Azure equivalent in scope).
- New top-level `edit_image` public API. The existing `ReqLLM.generate_image/3` is the single entry point.

## 3. Public API

Unchanged surface:

```elixir
# Text-to-image (generations)
ReqLLM.generate_image(
  "azure:gpt-image-1.5",
  "A photograph of a red fox in an autumn forest",
  base_url: "https://hello-mo8nnbxi-swedencentral.cognitiveservices.azure.com/openai",
  deployment: "gpt-image-1.5",
  size: "1024x1024",
  quality: :medium,
  output_format: :png,
  provider_options: [output_compression: 100]
)

# Image-to-image (edits) — input image lives in the context
input = File.read!("image_to_edit.png")
mask  = File.read!("mask.png")

context =
  ReqLLM.Context.new([
    ReqLLM.Context.user([
      ReqLLM.Message.ContentPart.text("Make this black and white"),
      ReqLLM.Message.ContentPart.image(input, "image/png")
    ])
  ])

ReqLLM.generate_image(
  "azure:gpt-image-1.5",
  context,
  base_url: "https://hello-mo8nnbxi-swedencentral.cognitiveservices.azure.com/openai",
  deployment: "gpt-image-1.5",
  provider_options: [mask: mask]
)
```

Response shape unchanged: `%ReqLLM.Response{message: %Message{role: :assistant, content: [%ContentPart{type: :image, data: <<…>>, media_type: "image/png"}, …]}}`. `ReqLLM.Response.image_data/1`, `image/1`, `images/1` continue to work.

## 4. Routing logic

A new clause is added to `ReqLLM.Providers.Azure`:

```elixir
def prepare_request(:image, model_spec, prompt_or_messages, opts) do
  with {:ok, model}         <- ReqLLM.model(model_spec),
       model_id             = effective_model_id(model),
       :ok                  <- validate_image_model(model_id),
       {:ok, context, prompt} <- image_context(prompt_or_messages, opts),
       {sub_op, image_parts} = classify_image_op(context),
       resolved_base_url    = resolve_base_url(get_model_family(model_id), opts),
       :ok                  <- reject_unsupported_endpoint_format(resolved_base_url),
       opts_with_context    = opts |> Keyword.put(:context, context) |> Keyword.put(:base_url, resolved_base_url),
       {:ok, processed_opts} <- ReqLLM.Provider.Options.process(__MODULE__, :image, model, opts_with_context),
       {api_version, deployment, base_url} = extract_azure_credentials(model, processed_opts) do
    build_image_request(sub_op, model, prompt, image_parts, processed_opts, deployment, api_version, base_url)
  end
end
```

- `validate_image_model/1` verifies the model family is `"gpt-image"` (added to `@model_families`). Any other family attempting `:image` yields `Invalid.Parameter`.
- `classify_image_op/1` inspects the last user message. If its content list contains any `%ContentPart{type: :image}`, returns `{:edit, image_parts}`; otherwise `{:generate, []}`.
- `reject_unsupported_endpoint_format/1` returns `{:error, …}` for Foundry or v1 GA base URLs on `:image` ops, pointing the user at the traditional format.

Longest-prefix match on `@model_families` already guarantees `"gpt-image"` wins over `"gpt"`. No change needed there beyond adding the entry.

## 5. Endpoint paths

```elixir
defp image_endpoint_path(:generate, deployment, api_version),
  do: "/deployments/#{deployment}/images/generations?api-version=#{api_version}"

defp image_endpoint_path(:edit, deployment, api_version),
  do: "/deployments/#{deployment}/images/edits?api-version=#{api_version}"
```

The existing chat-path builder is not reused — image paths are simple and image-specific, and keeping them in their own function avoids cluttering `get_chat_endpoint_path_by_family/4`.

## 6. Body encoding

A new module `ReqLLM.Providers.Azure.Images` owns body construction and response parsing, mirroring the shape of `Azure.OpenAI` / `Azure.Anthropic`.

### 6.1 Generations (JSON)

```elixir
def format_generate_request(_model_id, prompt, opts) do
  provider_opts = Keyword.get(opts, :provider_options, [])

  %{"prompt" => prompt, "n" => Keyword.get(opts, :n, 1)}
  |> maybe_put_size(Keyword.get(opts, :size))
  |> maybe_put_string("quality", Keyword.get(opts, :quality))
  |> maybe_put_string("output_format", Keyword.get(opts, :output_format))
  |> maybe_put_integer("output_compression", Keyword.get(provider_opts, :output_compression))
  |> maybe_put_string("background", Keyword.get(provider_opts, :background))
  |> maybe_put_string("moderation", Keyword.get(provider_opts, :moderation))
  |> maybe_put_string("user", Keyword.get(opts, :user))
end
```

No `"model"` field — deployment is in the URL path. `nil`-valued keys are omitted via the `maybe_put_*` helpers.

### 6.2 Edits (multipart)

Returned as a list of `{key, value}` tuples suitable for Req's `form_multipart:` option:

```elixir
def format_edit_request(_model_id, prompt, image_parts, opts) do
  provider_opts = Keyword.get(opts, :provider_options, [])

  image_fields =
    image_parts
    |> validate_image_parts!()
    |> Enum.map(fn %ContentPart{data: data, media_type: mt} ->
      {:image, {data, filename: "image.#{ext_for(mt)}", content_type: mt || "image/png"}}
    end)

  mask_field =
    case Keyword.get(provider_opts, :mask) do
      nil ->
        []
      bytes when is_binary(bytes) ->
        [{:mask, {bytes, filename: "mask.png", content_type: "image/png"}}]
      %ContentPart{type: :image, data: bytes, media_type: mt} ->
        [{:mask, {bytes, filename: "mask.#{ext_for(mt)}", content_type: mt || "image/png"}}]
    end

  image_fields ++
    mask_field ++
    [{:prompt, prompt}] ++
    maybe_multipart(:n, Keyword.get(opts, :n)) ++
    maybe_multipart(:size, Keyword.get(opts, :size)) ++
    maybe_multipart(:quality, Keyword.get(opts, :quality))
end
```

- `validate_image_parts!/1` raises `Invalid.Parameter` if any `ContentPart` has `type: :image_url` (URL only) or missing `data`. No URL-fetching.
- `ext_for/1` maps `"image/png" → "png"`, `"image/jpeg" → "jpg"`, `"image/webp" → "webp"`; defaults to `"png"`.
- The Req request is built with `form_multipart: parts` and **no explicit `content-type` header** so Req can set the multipart boundary.

### 6.3 Attach/auth

The image request shares the same auth resolution as chat (`resolve_api_key/3`, `build_auth_header/3`). Auth header name (`api-key` vs `authorization`) is selected the same way as today.

The current `attach/3` unconditionally sets `content-type: application/json`, which is wrong for multipart. The fix: thread a `:skip_content_type` flag through `Req.Request.put_private(request, :skip_content_type, true)` from the image-edit build path, and change `attach/3` to only put the JSON content-type header when that private flag is not set. For multipart, Req itself writes the correct `content-type` header (with boundary) during the `form_multipart` encoding step.

No other callers are affected — every existing operation path leaves the flag unset and therefore gets the JSON header as before.

## 7. Response parsing

`Azure.Images.parse_response/3` mirrors `OpenAI.ImagesAPI.decode_images_response/2`:

```elixir
def parse_response(body, model, opts) do
  data = Map.get(body, "data", [])
  media_type = media_type_for(Keyword.get(opts, :output_format))

  parts =
    data
    |> Enum.map(&decode_image_item(&1, media_type))
    |> Enum.reject(&is_nil/1)

  response = %ReqLLM.Response{
    id: image_response_id(),
    model: model.id,
    context: Keyword.get(opts, :context) || %ReqLLM.Context{messages: []},
    message: %ReqLLM.Message{role: :assistant, content: parts},
    stream?: false,
    finish_reason: :stop,
    usage: build_image_usage(parts, opts),
    provider_meta: %{"azure" => Map.delete(body, "data")}
  }

  {:ok, ReqLLM.Context.merge_response(response.context, response)}
end
```

`decode_image_item/2` handles both `b64_json` and `url` keys (Azure returns `b64_json` by default; url is defensive).

`Azure.decode_response/1` already dispatches to `formatter.parse_response(body, model, opts)` based on `Req.Request.get_private(request, :formatter)`. Setting `:formatter` to `Azure.Images` in the image prepare path is the only change needed — no modifications to the decode pipeline.

## 8. Error handling

| Condition | Behavior |
| --- | --- |
| Non-2xx response | Existing `Azure.decode_response/1` fallback extracts `error.message` / `error.code` → `ReqLLM.Error.API.Response`. No change. |
| Image ContentPart with only `url` (no `data`) | `Invalid.Parameter` with message directing user to `ContentPart.image/2` with binary data. |
| Foundry or v1 GA base URL + `:image` op | `Invalid.Parameter` pointing user to the traditional Azure OpenAI Service URL form `.../openai`. |
| Non-`gpt-image` model + `:image` op | `Invalid.Parameter`: "Azure model '<id>' does not support image operations. Use a gpt-image-* deployment." |
| Missing API key | Existing `resolve_api_key/3` raising path applies unchanged. |
| Mask provided but no input image | `Invalid.Parameter`: "mask requires at least one input image ContentPart in the user message." |

## 9. Testing

Unit tests in `test/provider/azure/azure_test.exs` (no live API calls):

1. **Generations routing.** `prepare_request(:image, "azure:gpt-image-1.5", "hello", …)` builds a `Req.Request` with `url` ending in `/deployments/gpt-image-1.5/images/generations?api-version=<default>` and JSON body containing `prompt`, `n`, `size`.
2. **Edits routing.** Context with a user message containing `[text, image]` parts triggers `/images/edits` path; request has `form_multipart` option set with the expected key set (`:image`, `:prompt`) and correct content types.
3. **Mask propagation.** `provider_options: [mask: mask_bytes]` inserts a `:mask` entry in the multipart list with `image/png` content-type by default.
4. **Auth header selection.** `api-key` header on plain api-key; `authorization: Bearer …` when api_key starts with `"Bearer "`. Tested for both sub-ops.
5. **Non-binary image rejection.** Image ContentPart with `type: :image_url` produces `{:error, %Invalid.Parameter{}}`.
6. **Unsupported endpoint formats.** `base_url: ".../services.ai.azure.com"` or `.../openai/v1` with `:image` op returns `{:error, %Invalid.Parameter{}}`.
7. **Wrong model family.** `prepare_request(:image, "azure:gpt-4o", …)` returns `{:error, %Invalid.Parameter{}}`.

No fixture/coverage test is added in this pass; the unit tests cover the request-shaping logic without hitting Azure. A manual smoke-test snippet is included in the PR description / guide update rather than as an automated test, since real Azure gpt-image deployments require per-user subscription.

## 10. Documentation

- `guides/azure.md` — new "Image generation" subsection showing both `generate_image` call shapes and noting the traditional-format-only constraint.
- `guides/image-generation.md` — remove "not yet supported" from the Azure note (if present) and add Azure alongside OpenAI/Google sections with a short example.

## 11. Out-of-scope follow-ups

- Port `/images/edits` support to the OpenAI provider (same body logic, different auth/base URL). Clean follow-up once this Azure work is in.
- Accept a `Path.t()` as convenience sugar for image ContentPart creation (e.g., `ContentPart.image_file("path.png")`). Doesn't belong in this change.
- Azure AI Foundry image-generation endpoint support, if/when Microsoft publishes one for gpt-image models.
