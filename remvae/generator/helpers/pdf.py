from fpdf import FPDF

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(left=25, top=20, right=25)

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, 'Training Report', ln=True, align='C')
        self.ln(5)

    def add_section_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 70, 140)
        self.cell(0, 10, title, ln=True)
        self.ln(4)

    def add_image(self, image_path, w=140):
        page_width = self.w - self.l_margin - self.r_margin
        x = self.l_margin + (page_width - w) / 2
        self.image(image_path, x=x, w=w)
        self.ln(10)

    def add_text(self, text):
        self.set_font('Arial', '', 12)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_table(self, data_dict):
        self.set_font('Arial', '', 12)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.set_draw_color(200, 200, 200)

        label_width = 60
        value_width = self.w - self.l_margin - self.r_margin - label_width
        line_height = 6

        for key, val in data_dict.items():
            key_text = key.replace('_', ' ').title()

            if isinstance(val, list):
                value_text = ", ".join(map(str, val))
            elif isinstance(val, int):
                value_text = str(val)
            else:
                try:
                    value_text = f"{float(val):.4f}"
                except (ValueError, TypeError):
                    value_text = str(val)

            x = self.get_x()
            y = self.get_y()

            self.set_font('Arial', 'B', 12)
            key_line_count = self.get_num_lines(key_text, label_width)

            self.set_font('Arial', '', 12)
            value_line_count = self.get_num_lines(value_text, value_width)

            max_lines = max(key_line_count, value_line_count)
            cell_height = max_lines * line_height

            self.set_fill_color(240, 240, 240)
            self.rect(x, y, label_width, cell_height, 'F')
            self.rect(x + label_width, y, value_width, cell_height)

            self.set_xy(x, y)
            self.set_font('Arial', 'B', 12)
            self.multi_cell(label_width, line_height, key_text, border=0)

            self.set_xy(x + label_width, y)
            self.set_font('Arial', '', 12)
            self.multi_cell(value_width, line_height, value_text, border=0)

            self.set_y(y + cell_height)

    def get_num_lines(self, text, width):
        if not text:
            return 1
        words = text.split()
        lines = 1
        current_line = ''
        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            if self.get_string_width(test_line) < width:
                current_line = test_line
            else:
                lines += 1
                current_line = word
        return lines


