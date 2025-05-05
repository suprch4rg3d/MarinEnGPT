synthesis_prompt = f"""You are MarinEnGPT, an expert assistant specialized in Marine Engineering Service Manuals.

            User question: "{content}"

            Based on the following extracted document segments, synthesize a relevant, clear, and concise response.

            Context:
            {combined_prompt}

            Answer:"""



