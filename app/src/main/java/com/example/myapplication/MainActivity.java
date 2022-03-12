package com.example.nirosh;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {

    TextView rec1;
    Button button1;
    TextView rec2;
    Button button2;
    TextView rec3;
    Button button3;
    TextView rec4;
    Button button4;
    Button button5;

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

                Intent intent = new Intent(MainActivity.this,MainActivity2.class);
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
                    String path = data.getData().getPath();
                    rec1.setText(path);
                }
                break;

            case 2:
                if (resultCode== RESULT_OK) {
                    String path = data.getData().getPath();
                    rec2.setText(path);
                }
                break;
            case 3:
                if (resultCode== RESULT_OK) {
                    String path = data.getData().getPath();
                    rec3.setText(path);
                }
                break;
            case 4:
                if (resultCode== RESULT_OK) {
                    String path = data.getData().getPath();
                    rec4.setText(path);
                }
                break;



        }

    }
}