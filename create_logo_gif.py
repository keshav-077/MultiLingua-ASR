"""
MultiLingua ASR Logo GIF Generator
Creates an animated logo with globe and language symbols
"""

from PIL import Image, ImageDraw, ImageFont
import math

def create_multilingua_logo():
    # Settings
    width, height = 800, 400
    frames = 60
    fps = 30
    
    # Colors (Green-Blue gradient theme)
    bg_color = (255, 255, 255)
    green = (16, 185, 129)     # #10b981
    blue = (6, 182, 212)       # #06b6d4
    purple = (139, 92, 246)    # #8b5cf6
    
    images = []
    
    for frame in range(frames):
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Animation progress
        progress = frame / frames
        angle = progress * 2 * math.pi
        
        # Center position
        center_x, center_y = width // 2, height // 2 - 20
        
        # Draw rotating outer ring
        ring_rotation = progress * 360
        for i in range(36):
            dot_angle = math.radians(i * 10 + ring_rotation)
            radius = 140
            x = center_x + radius * math.cos(dot_angle)
            y = center_y + radius * math.sin(dot_angle)
            
            opacity = int(100 + 155 * ((i % 12) / 12))
            color = green if i % 3 == 0 else (blue if i % 3 == 1 else purple)
            
            dot_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            dot_draw = ImageDraw.Draw(dot_img)
            dot_draw.ellipse([x-2, y-2, x+2, y+2], fill=(*color, opacity))
            img = Image.alpha_composite(img.convert('RGBA'), dot_img)
            img = img.convert('RGB')
        
        # Draw main globe circle
        globe_radius = 80
        pulse = math.sin(angle) * 3
        draw.ellipse(
            [center_x - globe_radius - pulse, center_y - globe_radius - pulse,
             center_x + globe_radius + pulse, center_y + globe_radius + pulse],
            outline=blue,
            width=3
        )
        
        # Draw latitude lines
        for lat in [-30, 0, 30]:
            lat_height = lat / 90 * globe_radius
            ellipse_rx = globe_radius * math.cos(math.radians(abs(lat)))
            draw.ellipse(
                [center_x - ellipse_rx, center_y + lat_height - 10,
                 center_x + ellipse_rx, center_y + lat_height + 10],
                outline=green,
                width=2
            )
        
        # Draw rotating longitude lines
        for i in range(4):
            long_angle = math.radians(i * 45 + ring_rotation)
            # Vertical ellipse rotated
            points = []
            for t in range(37):
                t_angle = math.radians(t * 10)
                x = center_x + 25 * math.cos(long_angle) * math.cos(t_angle)
                y = center_y + globe_radius * math.sin(t_angle)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=purple, width=2)
        
        # Draw orbiting language symbols
        languages = ['A', 'あ', 'ا', '中']
        for i, char in enumerate(languages):
            orbit_angle = angle + i * (2 * math.pi / len(languages))
            orbit_radius = 100
            char_x = center_x + orbit_radius * math.cos(orbit_angle)
            char_y = center_y + orbit_radius * math.sin(orbit_angle)
            
            try:
                char_font = ImageFont.truetype("arial.ttf", 28)
            except:
                char_font = ImageFont.load_default()
            
            color = green if i % 2 == 0 else blue
            
            # Get text size for centering
            bbox = draw.textbbox((0, 0), char, font=char_font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            draw.text(
                (char_x - char_width/2, char_y - char_height/2),
                char,
                fill=color,
                font=char_font
            )
        
        # Draw audio waveform in center
        bar_count = 6
        bar_spacing = 10
        total_width = bar_count * bar_spacing
        start_x = center_x - total_width // 2
        
        for i in range(bar_count):
            bar_x = start_x + i * bar_spacing
            # Each bar animates at different phase
            bar_progress = (progress + i * 0.15) % 1
            bar_height = max(6, 5 + 20 * math.sin(bar_progress * 2 * math.pi))
            
            color = green if i % 2 == 0 else blue
            y_top = center_y - bar_height/2
            y_bottom = center_y + bar_height/2
            
            # Ensure y1 >= y0
            if y_bottom < y_top:
                y_top, y_bottom = y_bottom, y_top
            
            draw.rounded_rectangle(
                [bar_x, y_top, bar_x + 6, y_bottom],
                radius=2,
                fill=color
            )
        
        # Add text
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Main title
        title = "MultiLingua ASR"
        text_bbox = draw.textbbox((0, 0), title, font=font_large)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height - 100
        
        # Gradient text effect
        draw.text((text_x, text_y), title, fill=green, font=font_large)
        
        # Subtitle
        subtitle = "Code-Switching Speech Recognition"
        sub_bbox = draw.textbbox((0, 0), subtitle, font=font_small)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_width) // 2
        draw.text((sub_x, text_y + 60), subtitle, fill=(100, 100, 100), font=font_small)
        
        images.append(img)
    
    # Save as GIF
    output_path = 'MultiLingua_ASR_Logo.gif'
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,
        loop=0,
        optimize=False
    )
    
    print(f"✅ Logo saved as: {output_path}")
    print(f"   Size: {width}x{height}")
    print(f"   Frames: {frames}")
    print(f"   FPS: {fps}")

if __name__ == "__main__":
    create_multilingua_logo()
