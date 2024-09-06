import { NextResponse } from "next/server";
import { BedrockAgentRuntimeClient, RetrieveAndGenerateCommand } from "@aws-sdk/client-bedrock-agent-runtime";

// Define the system prompt to generate case studies and questions
const systemPrompt = `You are an AI tasked with providing summarized case studies based on the user's input. 
Your task is as follows:
1. Summarize 10 case studies, each in about 100 words, based on the user's role, specialty, and department.
2. After each case study, generate 3 relevant and thought-provoking questions based on the core principles of the case study.
3. Ensure the summaries and questions are clear and concise, and tailored to the user's specific role and specialty.

Return the output in the following JSON format:
{
    "caseStudies": [
        {
            "summary": str,
            "questions": [
                str, str, str
            ]
        }
    ]
}`;

export async function POST(req: Request): Promise<Response> {
    // Initialize the Bedrock Agent Runtime client
    const client = new BedrockAgentRuntimeClient({ region: "us-east-1" });

    try {
        // Parse the request body
        const body = await req.json();
        const { role, specialty, department } = body;  // Destructure without type annotations
        console.log(`Received role: ${role}, specialty: ${specialty}, department: ${department}`);

        if (!role || !specialty || !department) {
            return new Response(JSON.stringify({ error: "Role, specialty, and department are required" }), {
                status: 400,
                headers: { 'Content-Type': 'application/json' },
            });
        }

        // Define the input for the RetrieveAndGenerateCommand
        const input: any = {
            input: { text: `Role: ${role}, Specialty: ${specialty}, Department: ${department}` },
            retrieveAndGenerateConfiguration: {
                type: "KNOWLEDGE_BASE",
                knowledgeBaseConfiguration: {
                    knowledgeBaseId: "TO2LMBBASW", // Replace with your actual Knowledge Base ID
                    modelArn: "anthropic.claude-3-haiku-20240307-v1:0", // Replace with your model ARN
                    retrievalConfiguration: {
                        vectorSearchConfiguration: {
                            numberOfResults: 10, // Number of case studies to retrieve
                            overrideSearchType: "SEMANTIC"
                        }
                    },
                    generationConfiguration: {
                        promptTemplate: {
                            textPromptTemplate: `${systemPrompt} $search_results$`
                        },
                        inferenceConfig: {
                            textInferenceConfig: {
                                temperature: 0.7,
                                topP: 0.9,
                                maxTokens: 2048 // Allow larger tokens to include multiple case studies and questions
                            }
                        },
                    },
                    orchestrationConfiguration: {
                        queryTransformationConfiguration: {
                            type: "QUERY_DECOMPOSITION",
                        }
                    }
                }
            }
        };

        // Create the command
        const command = new RetrieveAndGenerateCommand(input);

        // Send the command to Bedrock and wait for the response
        const response = await client.send(command);

        // Extract the response text and format it
        const caseStudies = JSON.parse(response.output?.text || "No response from model");

        // Return the case studies and questions as a JSON response
