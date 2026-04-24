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
  # alias ReqLLM.Context
  # alias ReqLLM.Message
  # alias ReqLLM.Message.ContentPart
  # alias ReqLLM.Response

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

  defp maybe_put_string(body, _key, nil), do: body

  defp maybe_put_string(body, key, value) when is_atom(value),
    do: Map.put(body, key, Atom.to_string(value))

  defp maybe_put_string(body, key, value) when is_binary(value), do: Map.put(body, key, value)

  defp maybe_put_integer(body, _key, nil), do: body

  defp maybe_put_integer(body, key, value) when is_integer(value), do: Map.put(body, key, value)

  @doc "Build the multipart parts list for /images/edits."
  def format_edit_request(_model_id, prompt, image_parts, _opts) do
    {prompt, image_parts}
  end

  @doc "Parse an image response body into a ReqLLM.Response."
  def parse_response(_body, _model, _opts), do: {:error, :not_implemented}

  @doc "Map an :output_format option to the corresponding MIME type. Overridden in Task 3."
  def media_type_for(_format), do: "image/png"

  @doc "Validate a list of image ContentParts for use in a multipart edit. Overridden in Task 6."
  def validate_image_parts!(parts), do: parts
end
