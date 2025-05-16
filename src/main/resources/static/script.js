async function analyzeText() {
    const textInput = document.getElementById('textInput').value;
    const emojiResult = document.getElementById('emojiResult');
    
    if (!textInput.trim()) {
        alert('Bitte geben Sie einen Text ein!');
        return;
    }
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain'
            },
            body: textInput
        });
        
        if (!response.ok) {
            throw new Error('Fehler bei der Analyse');
        }
        
        const emoji = await response.text();
        emojiResult.textContent = emoji;
    } catch (error) {
        console.error('Fehler:', error);
        emojiResult.textContent = '❌';
        alert('Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.');
    }
} 