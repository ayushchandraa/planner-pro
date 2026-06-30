import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy initialization of Gemini client
let aiClient: GoogleGenAI | null = null;
function getAI(): GoogleGenAI | null {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return null;
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", apiConfigured: !!process.env.GEMINI_API_KEY });
});

// 1. Analyze Schedule API
app.post("/api/gemini/analyze-schedule", async (req, res) => {
  const { tasks, calendarEvents, habits, workSettings, currentTime } = req.body;
  const ai = getAI();

  if (!ai) {
    // High fidelity fallback when GEMINI_API_KEY is not configured
    return res.json(getFallbackScheduleAnalysis(tasks, calendarEvents, habits, currentTime));
  }

  try {
    const prompt = `Analyze this user's current productivity schedule.
Tasks list: ${JSON.stringify(tasks)}
Calendar events (fixed commitments): ${JSON.stringify(calendarEvents)}
Habits to complete: ${JSON.stringify(habits)}
Work Settings (focus hours, start/end times): ${JSON.stringify(workSettings)}
Current Time: ${currentTime}

Perform an AI analysis of their schedule.
- Predict the deadline miss risk (0 to 100).
- Give a short explanation for the risk.
- Determine the absolute "Next Best Action" (one of the tasks or habits).
- Suggest an optimal sequencing of task IDs.
- Produce a day schedule slots sequence from start to end of day.
- Provide 3 tailored, context-aware productivity recommendations.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            riskScore: {
              type: Type.INTEGER,
              description: "Productivity risk score from 0 (relaxed, on track) to 100 (extreme danger of missed deadlines)."
            },
            riskAnalysis: {
              type: Type.STRING,
              description: "Brief human-friendly explanation of why they are at risk or on track."
            },
            nextBestAction: {
              type: Type.STRING,
              description: "The title of the single best task/habit to work on next."
            },
            recommendedSequence: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Ordered list of task IDs to tackle in optimal order."
            },
            scheduleSlots: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  time: { type: Type.STRING, description: "e.g., '09:00 AM' or '14:30'" },
                  type: { type: Type.STRING, description: "Must be 'task' or 'habit' or 'commitment' or 'rest'" },
                  label: { type: Type.STRING, description: "Display name for this slot" },
                  durationMinutes: { type: Type.INTEGER, description: "Duration in minutes" },
                  taskId: { type: Type.STRING, description: "Optional associated task ID" }
                },
                required: ["time", "type", "label", "durationMinutes"]
              },
              description: "Full sequential timeline representing the day's optimized planner slots."
            },
            recommendations: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "3 concrete, actionable recommendations based on past performance, focus hours, and fatigue."
            }
          },
          required: ["riskScore", "riskAnalysis", "nextBestAction", "recommendedSequence", "scheduleSlots", "recommendations"]
        }
      }
    });

    const resultText = response.text || "{}";
    const parsed = JSON.parse(resultText);
    res.json(parsed);
  } catch (error: any) {
    console.error("Gemini Analyze Schedule Error:", error);
    res.status(500).json({ error: "Failed to analyze schedule with AI.", details: error.message });
  }
});

// 2. SOS Rescue Mode API
app.post("/api/gemini/rescue-mode", async (req, res) => {
  const { tasks, calendarEvents, currentTime } = req.body;
  const ai = getAI();

  if (!ai) {
    return res.json(getFallbackRescuePlan(tasks, calendarEvents));
  }

  try {
    const prompt = `SOS! The user is falling behind and triggered Rescue Mode.
Tasks: ${JSON.stringify(tasks)}
Calendar commitments: ${JSON.stringify(calendarEvents)}
Current Time: ${currentTime}

Analyze what is critical, what can be deferred/delayed, and create a high-impact recovery plan.
- Draft 1-2 email requests for meeting rescheduling or extension requests.
- Provide concrete step-by-step actions to recover.
- Suggest task IDs to defer to tomorrow or later.
- Provide a calming, high-focus productivity quote.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            recoverySteps: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Clear, sequential tactical steps to execute immediately. Highly focused."
            },
            deferredTasks: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "List of Task IDs recommended to defer to a later date to clear cognitive overload."
            },
            extensionEmails: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  subject: { type: Type.STRING },
                  recipient: { type: Type.STRING, description: "Suggested recipient placeholder e.g. 'Project Lead', 'Manager', 'Professor'" },
                  body: { type: Type.STRING },
                  reason: { type: Type.STRING, description: "Why we are postponing this" }
                },
                required: ["subject", "recipient", "body", "reason"]
              },
              description: "Drafted communication templates to immediately request more time."
            },
            calmingQuote: {
              type: Type.STRING,
              description: "A calming yet motivating focus quote."
            }
          },
          required: ["recoverySteps", "deferredTasks", "extensionEmails", "calmingQuote"]
        }
      }
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText));
  } catch (error: any) {
    console.error("Gemini Rescue Mode Error:", error);
    res.status(500).json({ error: "Failed to generate Rescue Plan.", details: error.message });
  }
});

// 3. Draft Email API
app.post("/api/gemini/draft-email", async (req, res) => {
  const { taskTitle, dueDate, reason, recipientName, tone } = req.body;
  const ai = getAI();

  if (!ai) {
    return res.json({
      subject: `Rescheduling request: ${taskTitle || 'Our discussion'}`,
      body: `Hi ${recipientName || 'there'},\n\nI am writing to request a short extension/reschedule for "${taskTitle || 'our task'}". Due to ${reason || 'unforeseen schedule conflicts'}, I would like to propose adjusting the deadline of ${dueDate || 'today'} slightly.\n\nThank you for your understanding.\n\nBest regards,\n[Your Name]`
    });
  }

  try {
    const prompt = `Draft an email requesting a deadline extension or a meeting reschedule.
Task/Meeting: ${taskTitle}
Current Due Date/Time: ${dueDate}
Reason: ${reason}
Recipient Name: ${recipientName}
Tone requested: ${tone} (e.g. professional, urgent, casual)`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            subject: { type: Type.STRING },
            body: { type: Type.STRING }
          },
          required: ["subject", "body"]
        }
      }
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText));
  } catch (error: any) {
    console.error("Gemini Draft Email Error:", error);
    res.status(500).json({ error: "Failed to draft email.", details: error.message });
  }
});

// 4. Break Down Goal API
app.post("/api/gemini/break-down-goal", async (req, res) => {
  const { goalTitle, description, totalHours } = req.body;
  const ai = getAI();

  if (!ai) {
    // Fallback subtasks
    return res.json({
      subtasks: [
        { title: "Define requirements & outline scope", durationMinutes: 60, difficulty: "easy" },
        { title: "Execute core development / main task draft", durationMinutes: 120, difficulty: "hard" },
        { title: "Refining & detailing aspects", durationMinutes: 90, difficulty: "medium" },
        { title: "Final testing and review", durationMinutes: 45, difficulty: "easy" }
      ]
    });
  }

  try {
    const prompt = `Break down the following long-term goal or large task into actionable subtasks.
Goal: ${goalTitle}
Description: ${description || "None provided"}
Target Time budget (if any): ${totalHours ? totalHours + " hours" : "Not specified"}

Generate a list of 3-6 sequential subtasks with title, estimated minutes, difficulty, and optional prior dependencies.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            subtasks: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  durationMinutes: { type: Type.INTEGER },
                  difficulty: { type: Type.STRING, description: "easy, medium, hard" },
                  dependency: { type: Type.STRING, description: "Title of previous subtask this depends on (optional)" }
                },
                required: ["title", "durationMinutes", "difficulty"]
              }
            }
          },
          required: ["subtasks"]
        }
      }
    });

    res.json(JSON.parse(response.text || "{}"));
  } catch (error: any) {
    console.error("Gemini Break Down Goal Error:", error);
    res.status(500).json({ error: "Failed to break down goal.", details: error.message });
  }
});

// 5. Voice/Natural Language Parsing API
app.post("/api/gemini/voice-parse", async (req, res) => {
  const { transcript } = req.body;
  const ai = getAI();

  if (!ai) {
    return res.json({
      tasks: [
        {
          title: transcript.substring(0, 50) || "Voice Captured Task",
          urgency: "high",
          importance: "medium",
          durationMinutes: 45,
          subtasks: ["Review constraints", "Take immediate action"]
        }
      ]
    });
  }

  try {
    const prompt = `The user said the following verbal transcript to request task creation or planning:
"${transcript}"

Intelligently parse this natural language request.
Extract 1 or more structured tasks, determining:
- task title (concise, clear, e.g. "Draft budget report")
- urgency ('high', 'medium', 'low')
- importance ('high', 'medium', 'low')
- estimated duration in minutes
- estimated due date (if specified, relative to the current year 2026)
- nested subtasks (actionable sub-steps to complete the task)`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            tasks: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  urgency: { type: Type.STRING, description: "high, medium, low" },
                  importance: { type: Type.STRING, description: "high, medium, low" },
                  durationMinutes: { type: Type.INTEGER },
                  dueDate: { type: Type.STRING, description: "YYYY-MM-DD format (if specified or implied)" },
                  subtasks: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING }
                  }
                },
                required: ["title", "urgency", "importance", "durationMinutes"]
              }
            }
          },
          required: ["tasks"]
        }
      }
    });

    res.json(JSON.parse(response.text || "{}"));
  } catch (error: any) {
    console.error("Gemini Voice Parse Error:", error);
    res.status(500).json({ error: "Failed to parse natural language.", details: error.message });
  }
});

// FALLBACK ANALYZER
function getFallbackScheduleAnalysis(tasks: any[], calendarEvents: any[], habits: any[], currentTime: string) {
  // Simple heuristic risk score
  let riskScore = 15;
  const highPriorityTasks = tasks.filter(t => t.urgency === "high" || t.importance === "high");
  const uncompletedHighPriority = highPriorityTasks.filter(t => !t.completed);
  riskScore += uncompletedHighPriority.length * 15;
  if (tasks.length > 6) riskScore += 10;
  if (calendarEvents.length > 4) riskScore += 10;
  riskScore = Math.min(riskScore, 100);

  let riskAnalysis = "Everything looks steady. Keep following the recommended slots.";
  if (riskScore > 60) {
    riskAnalysis = "High risk! You have multiple high-importance tasks due soon with tight calendar overlaps.";
  } else if (riskScore > 35) {
    riskAnalysis = "Moderate risk. A few upcoming deadlines require focus, but rescheduling non-essential items can buy you time.";
  }

  const nextTask = tasks.find(t => !t.completed) || habits.find(h => h.streak === 0) || { title: "Plan upcoming week" };

  const recSequence = tasks.filter(t => !t.completed)
    .sort((a, b) => {
      const aVal = (a.urgency === 'high' ? 2 : 1) + (a.importance === 'high' ? 2 : 1);
      const bVal = (b.urgency === 'high' ? 2 : 1) + (b.importance === 'high' ? 2 : 1);
      return bVal - aVal;
    })
    .map(t => t.id);

  // Generate some realistic slots
  const scheduleSlots = [];
  let currentHour = 9;
  
  // Mix events, habits, and tasks
  calendarEvents.forEach(e => {
    scheduleSlots.push({
      time: e.time,
      type: "commitment",
      label: e.title,
      durationMinutes: e.duration || 60
    });
  });

  tasks.filter(t => !t.completed).slice(0, 3).forEach((t, idx) => {
    scheduleSlots.push({
      time: `${String(currentHour + idx + 1).padStart(2, '0')}:00 AM`,
      type: "task",
      label: t.title,
      durationMinutes: t.durationMinutes || 60,
      taskId: t.id
    });
  });

  habits.slice(0, 2).forEach((h, idx) => {
    scheduleSlots.push({
      time: `08:${30 + idx * 15} AM`,
      type: "habit",
      label: h.title,
      durationMinutes: 15
    });
  });

  // Sort slots by time roughly
  scheduleSlots.sort((a, b) => {
    return a.time.localeCompare(b.time);
  });

  return {
    riskScore,
    riskAnalysis,
    nextBestAction: nextTask.title,
    recommendedSequence: recSequence,
    scheduleSlots,
    recommendations: [
      "Review upcoming High Urgency tasks first to eliminate immediate bottlenecks.",
      "Integrate fixed calendar commitments prior to planning flexible study/work blocks.",
      "Schedule deep work blocks during your peak focus window (e.g., 9:00 AM - 11:30 AM)."
    ]
  };
}

// FALLBACK RESCUE PLAN
function getFallbackRescuePlan(tasks: any[], calendarEvents: any[]) {
  const atRisk = tasks.filter(t => !t.completed && (t.urgency === 'high' || t.importance === 'high'));
  const deferCandidates = tasks.filter(t => !t.completed && (t.urgency !== 'high' && t.importance !== 'high'));

  return {
    recoverySteps: [
      "Switch off phone notifications and block social media for 90 minutes.",
      "Execute a 25-minute Pomodoro focusing exclusively on: " + (atRisk[0]?.title || "your highest priority task"),
      "Reach out to team members to push back low-urgency discussion slots.",
      "Take a 5-minute stretch break before diving into your next critical deliverable."
    ],
    deferredTasks: deferCandidates.map(t => t.id),
    extensionEmails: [
      {
        subject: `Rescheduling Request: Extension for ${atRisk[0]?.title || 'our commitment'}`,
        recipient: "Project Lead / Client",
        reason: "unexpected calendar overlaps and prioritized urgent dependencies",
        body: `Hi there,\n\nI hope you're having a good day. Due to unexpected schedule overlaps today, I am working on resolving a critical bottleneck that has arisen.\n\nTo ensure the delivery is of the highest standard, would it be possible to adjust the timeline for "${atRisk[0]?.title || 'our task'}" slightly? Proposing a short 24-hour extension.\n\nThank you for your understanding and support.\n\nBest,\n[Your Name]`
      }
    ],
    calmingQuote: "Do not fear delays. Take a single deep breath, focus on the absolute next step, and execute calmly."
  };
}

// Vite and Express server setup
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Planner Pro (Last-Minute Life Saver) Server running on http://localhost:${PORT}`);
  });
}

startServer();
