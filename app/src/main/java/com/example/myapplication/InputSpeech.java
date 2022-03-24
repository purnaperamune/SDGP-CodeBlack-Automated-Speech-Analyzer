package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


public class InputSpeech extends AppCompatActivity {

    TextView rec1;
    Button button1;
    TextView rec2;
    Button button2;
    TextView rec3;
    Button button3;
    TextView rec4;
    Button button4;
    Button button5;
    String path="No recordings";
    String path1="No recordings";
    String path2="No recordings";
    String path3="No recordings";

    public static final int PICK_FILE1 =1;
    public static final int PICK_FILE2 =2;
    public static final int PICK_FILE3 =3;
    public static final int PICK_FILE4 =4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button1 = findViewById(R.id.button1);
        button2 = findViewById(R.id.button2);
        button3 = findViewById(R.id.button3);
        button4 = findViewById(R.id.button4);
        button5 = findViewById(R.id.rate);
        rec1=findViewById(R.id.rec1);
        rec2=findViewById(R.id.rec2);
        rec3=findViewById(R.id.rec3);
        rec4=findViewById(R.id.rec4);

        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("audio/*");
                startActivityForResult(intent, PICK_FILE1);

            }
        });
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("audio/*");
                startActivityForResult(intent, PICK_FILE2);
            }
        });
        button3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("audio/*");
                startActivityForResult(intent, PICK_FILE3);
            }
        });
        button4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("audio/*");
                startActivityForResult(intent, PICK_FILE4);
            }
        });

        button5.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {

                Intent intent = new Intent(InputSpeech.this, Feedback.class);
                intent.putExtra("message_key", path);
                intent.putExtra("message_key1", path1);
                intent.putExtra("message_key2", path2);
                intent.putExtra("message_key3", path3);
                startActivity(intent);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case 1:
                if (resultCode== RESULT_OK) {
                    path = data.getData().getPath();
                    rec1.setText(path);
                }
                break;

            case 2:
                if (resultCode== RESULT_OK) {
                    path1 = data.getData().getPath();
                    rec2.setText(path1);
                }
                break;
            case 3:
                if (resultCode== RESULT_OK) {
                    path2 = data.getData().getPath();
                    rec3.setText(path2);
                }
                break;
            case 4:
                if (resultCode== RESULT_OK) {
                    path3 = data.getData().getPath();
                    rec4.setText(path3);
                }
                break;




        }

    }
}