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
