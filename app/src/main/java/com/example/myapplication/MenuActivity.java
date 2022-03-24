package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;

public class MenuActivity extends AppCompatActivity {
    private NavController navController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

    }
    public void btnRateSpeech(View view) {
        Intent i = new Intent(this, InputSpeech.class);
        startActivity(i);
    }

    public void recorderPage(View view) {
        Intent i = new Intent(this,Recorder.class);
        startActivity(i);
    }

    public void btnPastSpeech(View view) {
        /*
        Should display the page of the past recordings with a small audio player at the
        bottom of the screen. Please use recylerView.
        */
        Intent intent= new Intent(MenuActivity.this, HistoryActivity.class);
        startActivity(intent);

    }
}