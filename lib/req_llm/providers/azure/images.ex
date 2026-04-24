defmodule ReqLLM.Providers.Azure.Images do
  @moduledoc """
  Image generation and edit formatter for the Azure provider.

  Covers Azure OpenAI Service deployments of gpt-image-* models:
    POST /openai/deployments/{deployment}/images/generations (JSON)
    POST /openai/deployments/{deployment}/images/edits       (multipart)

  Selection between the two endpoints is made by `ReqLLM.Providers.Azure`
  based on the content of the normalized context.
  """

  # Future aliases — uncomment as formatter bodies are filled in across Tasks 2–9:
  alias ReqLLM.Context
  alias ReqLLM.Message
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Response

  @doc "Build the JSON body for /images/generations."
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

  defp maybe_put_size(body, _), do: body

  defp maybe_put_string(body, _key, nil), do: body

  defp maybe_put_string(body, key, value) when is_atom(value),
    do: Map.put(body, key, Atom.to_string(value))

  defp maybe_put_string(body, key, value) when is_binary(value), do: Map.put(body, key, value)

  defp maybe_put_string(body, _key, _value), do: body

  defp maybe_put_integer(body, _key, nil), do: body

  defp maybe_put_integer(body, key, value) when is_integer(value), do: Map.put(body, key, value)

  defp maybe_put_integer(body, _key, _value), do: body

  @doc "Build the multipart parts list for /images/edits."
  def format_edit_request(_model_id, prompt, image_parts, _opts) do
    {prompt, image_parts}
  end

  @doc "Parse an image response body into a ReqLLM.Response."
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

  @doc "Map an :output_format option to the corresponding MIME type."
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

  @doc "Validate a list of image ContentParts for use in a multipart edit. Overridden in Task 6."
  def validate_image_parts!(parts), do: parts
end
