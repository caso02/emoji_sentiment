package com.emoji.controller;

import com.emoji.service.EmojiService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class EmojiController {

    private final EmojiService emojiService;

    public EmojiController(EmojiService emojiService) {
        this.emojiService = emojiService;
    }

    @GetMapping("/emoji")
    public String getEmoji(@RequestParam String text) {
        return emojiService.analyzeText(text);
    }
}
