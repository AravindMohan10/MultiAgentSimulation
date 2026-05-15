// Structured agent action — mirrors src/agents/schema.py LLMActionOutput contract.

enum AgentIntent {
  PursueTarget @description("Move toward or intercept a target")
  PursueMemory @description("Move toward last known position")
  FleeThreat @description("Move away from threats")
  HoldPosition
  ExploreArea
  SearchSystematic
  Regroup
  SignalTeammates
  CutOff
  HoldChokepoint
  Bait
  Unknown
}

class CoordMessage {
  payload float[]
  channel string?
}

class AgentAction {
  intent AgentIntent
  target_position float[]? @description("World [x, y]; preferred over movement")
  target_description string?
  confidence float
  movement float[]? @description("Deprecated [dx, dy, dz]; null when using target_position")
  message CoordMessage?
}

// Same prompts as prompts.py — passed through so before/after use identical instructions.
function ChooseAgentAction(system_prompt: string, user_prompt: string) -> AgentAction {
  client GroqLlama
  prompt #"
    {{ _.role("system") }}
    {{ system_prompt }}

    {{ _.role("user") }}
    {{ user_prompt }}

    {{ ctx.output_format }}
  "#
}
