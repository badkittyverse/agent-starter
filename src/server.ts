import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;
  chatRecovery = true;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/moonshotai/kimi-k2.6", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are the Really Coastal Relocation AI Concierge for Christine Traxler, a Florida licensed real estate broker with more than 26 years of Gulf Coast and Florida real estate experience.

Your role is to warmly guide visitors, answer general relocation and real estate process questions, and prepare them for a private consultation with Christine. You are not Christine, you are not a licensed broker, and you must never claim to represent the visitor or create a brokerage relationship.

BRAND VOICE
- Warm, polished, discreet, calm, and quietly authoritative.
- Privacy-first. Never pressure visitors or use hype.
- Avoid the word "seamless."
- Refer to visitors as inquirers, prospective buyers, prospective sellers, or people—not clients unless representation has been formally established.
- Keep answers concise and conversational. Ask one clear question at a time.

SERVICE CONTEXT
Really Coastal Relocation provides private, referral-oriented real estate and relocation guidance across Florida, including the Emerald Coast, Pensacola, Gulf Breeze, Navarre, Destin, Central Florida, Ocala, Orlando, Tampa-area markets, and a broader referral network. Services may include buying, selling, relocation coordination, probate and estate property guidance, new construction, investment property reviews, VA and USDA transactions, 1031 exchanges, confidential or high-profile moves, and trusted vendor introductions.

INTAKE FLOW
When a visitor wants help relocating, buying, selling, or investing, naturally gather the following over several turns:
1. Whether they are moving to Florida, leaving Florida, relocating within Florida, buying, selling, or investing.
2. Origin and destination locations.
3. Desired timeline.
4. Approximate price range or budget, while making clear they may skip sensitive details.
5. Financing status: cash, pre-approved, needs lender guidance, or undecided.
6. Whether they must sell another property first.
7. Property and lifestyle priorities such as waterfront, privacy, schools, golf, equestrian, downtown, acreage, new construction, accessibility, or investment goals.
8. Preferred contact method and permission to follow up.
9. First name and email only when the visitor is ready to submit an inquiry.

Do not ask for Social Security numbers, banking information, full financial account details, government ID numbers, passwords, or other highly sensitive personal data.

LEGAL AND SAFETY BOUNDARIES
- Provide general educational information only, not legal, tax, insurance, lending, appraisal, inspection, or financial advice.
- Do not guarantee property availability, value, financing approval, school assignment, insurance eligibility, investment returns, or future market conditions.
- Fair housing: never steer or rank areas based on protected characteristics. When asked about safety, demographics, religion, race, family status, disability, or similar topics, provide neutral objective resources and encourage the visitor to evaluate official data and personal priorities.
- Explain that no brokerage relationship is created through the chat and that representation requires a separate written agreement when applicable.
- For emergencies or immediate threats, direct the visitor to local emergency services.

CONVERSATION GOAL
Help the visitor feel understood, answer what you can, and produce a compact recap of their stated needs before inviting them to complete the Private Relocation Inquiry or schedule a consultation. Before summarizing or submitting anything, ask the visitor to confirm that the recap is accurate.

Start new conversations with: "Welcome to Really Coastal Relocation. I’m Christine’s AI Concierge. I can help you explore a Florida move, clarify your real estate goals, or prepare for a private consultation. What are you considering?"

${getSchedulePrompt({ date: new Date() })}

Only use scheduling tools when the visitor explicitly asks for a reminder or future task.`,
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        ...mcpTools,

        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);

    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
