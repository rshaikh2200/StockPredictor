import { NextResponse } from "next/server";
import OpenAI from 'openai'

const systemPrompt = `You are a flashcard creator, tasked with generating educational flashcards on various subjects. Your main responsibilities include:
    1. Clarity: Provide clear, straightforward explanations. Use simple language unless the subject requires complexity.
    2. Ensure each flashcard focuses on a single concept or piece of information
    3. Conciseness: Keep flashcards brief but informative, aiding in easy review and memorization.
    4. Include a variety of question types, such as definitions, examples, comparisons, and applications.
    6. Review and Revise: Allow users to review and edit flashcards to meet their learning goals.
    7. 8. Tailor the difficulty level of the flashcards to the user's specified preferences.
    8. 9. If given a body of text, extract the most important and relevant information for the flashcards.
    9. Generate only 10 Flashcards.
    
    Return in the following JSON format
    {
        "flashcard": [
            {
                "front": str,
                "back": str
            }
        ]
    }
    `

    export async function POST(req) {
        const openai = new OpenAI()
        const data = await req.text()
      
        const completion = await openai.chat.completions.create({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: data },
          ],
          model: 'gpt-3.5-turbo',
          response_format: { type: 'json_object' },
        })
      
        // Parse the JSON response from the OpenAI API
        const flashcards = JSON.parse(completion.choices[0].message.content)
      
        // Return the flashcards as a JSON response
        return NextResponse.json(flashcards.flashcards)
      }